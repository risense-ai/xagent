from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Annotated, Any, Callable, Iterator, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field, field_validator
from starlette import convertors as starlette_url  # codespell:ignore

from ...core.memory.base import MemoryStore
from ...core.memory.core import MemoryNote
from ...core.memory.remote import (
    READ_SCOPE_FILTER_KEY,
    RemoteMemoryNote,
    RemoteMemoryStore,
    RemoteMemoryStoreError,
)
from ..auth_dependencies import get_current_user
from ..dynamic_memory_store import get_memory_store, get_memory_store_manager
from ..memory_lifecycle import MemoryUnavailableError
from ..models.user import User
from ..revocable_memory_store import unwrap_memory_store
from ..services.db_runtime import run_db_io_cancellation_safe
from ..tracking.standalone_usage import usage_scope
from ..user_isolated_memory import UserContext

logger = logging.getLogger(__name__)
MEMORY_READ_UNAVAILABLE_DETAIL = "Memory storage is temporarily unavailable."
MEMORY_NOT_FOUND_DETAIL = "not found or not accessible"


# Memory IDs are opaque strings. New notes get ``str(uuid.uuid4())``, but the
# store keeps any non-empty ID it is given and legacy admission preserves
# existing unique non-empty IDs, so the by-ID routes must accept exactly what
# ``/list`` returned -- quotes, spaces, ``%``/``_`` and an encoded ``/``
# included. The value is only ever compared as an exact literal: every LanceDB
# predicate built from it goes through ``xagent.core.utils.lancedb_sql``. Only
# the empty string is refused, because no store can hold it (an empty ID is
# replaced on add and rejected by legacy admission).
class _MemoryIdConverter(starlette_url.Convertor[str]):  # codespell:ignore
    """One non-empty path remainder, ``/`` included.

    Unlike Starlette's ``path`` converter it never matches an empty remainder,
    so ``/api/memory/`` still falls through to the collection routes instead of
    answering 405 from a by-ID route. The pattern is DOTALL: Starlette compiles
    route regexes without it and ``$`` can match before a terminal newline, so a
    plain ``.+`` would route ``note\n`` to ``note``.
    """

    regex = r"(?s:.+)"

    def convert(self, value: str) -> str:
        return value

    def to_string(self, value: str) -> str:
        return value


starlette_url.register_url_convertor(  # codespell:ignore
    "memory_id", _MemoryIdConverter()
)

MemoryIdPath = Annotated[
    str,
    Path(min_length=1, description="Memory ID, exactly as listed"),
]
MEMORY_ID_ROUTE = "/{memory_id:memory_id}"


@contextmanager
def memory_operation() -> Iterator[None]:
    """Answer a revocation that lands mid-operation with the same stable 503.

    :attr:`MemoryManagementRouter.memory_store` maps only the *initial*
    acquisition failure. The store it returns is a revocable proxy that
    re-checks its publication generation on every call, so a concurrent
    revalidation -- an administrator changing the authority's vector space,
    another worker revoking the publication -- can raise
    :class:`MemoryUnavailableError` from the delegated operation instead,
    after the acquisition already succeeded.

    Without this, each route's broad ``except Exception`` turned that into a
    500 naming the internal failure. One shared boundary keeps every route on
    the single public-safe detail, and keeps the seven of them from drifting
    apart. ``store-info`` deliberately does not use it: it reports the
    lifecycle rather than operating on the store, and answers 200 throughout.
    """
    try:
        yield
    except MemoryUnavailableError as error:
        raise HTTPException(status_code=503, detail=error.status.detail) from None
    except RemoteMemoryStoreError as error:
        # A dependency failure (including an incomplete list) must never look
        # like a successful operation. Only the service's conflict is public
        # business state; other failures keep credentials and upstream details
        # behind the same stable unavailable response used by memory reads.
        if error.status_code == 409:
            raise HTTPException(status_code=409, detail=str(error)) from None
        raise HTTPException(
            status_code=503, detail=MEMORY_READ_UNAVAILABLE_DETAIL
        ) from None


class MemoryListRequest(BaseModel):
    category: Optional[str] = Field(None, description="Filter by memory category")
    tags: Optional[list[str]] = Field(
        None, description="Filter by tags (all must match)"
    )
    keywords: Optional[list[str]] = Field(
        None, description="Filter by keywords (all must match)"
    )
    date_from: Optional[datetime] = Field(
        None, description="Filter memories from this date"
    )
    date_to: Optional[datetime] = Field(
        None, description="Filter memories to this date"
    )
    limit: Optional[int] = Field(
        100, description="Maximum number of memories to return"
    )
    offset: Optional[int] = Field(0, description="Offset for pagination")


class MemoryUpdateRequest(BaseModel):
    content: Optional[str] = Field(None, description="Updated memory content")
    space: Optional[str] = None
    space_kind: Optional[str] = None
    space_id: Optional[int] = None
    keywords: Optional[list[str]] = Field(None, description="Updated memory keywords")
    tags: Optional[list[str]] = Field(None, description="Updated memory tags")
    category: Optional[str] = Field(None, description="Updated memory category")
    metadata: Optional[dict[str, Any]] = Field(
        None, description="Updated memory metadata"
    )

    @field_validator("content")
    @classmethod
    def reject_blank_content(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("Content must not be blank")
        return value


class MemoryListResponse(BaseModel):
    memories: list[dict[str, Any]]
    total_count: int
    filters_used: dict[str, Any]


class MemoryStatsResponse(BaseModel):
    total_count: int
    category_counts: dict[str, int]
    tag_counts: dict[str, int]
    memory_store_type: str
    error: Optional[str] = None


class MemoryManagementRouter:
    def __init__(
        self, memory_store_provider: Optional[Callable[[], MemoryStore]] = None
    ) -> None:
        """
        Initialize memory management router.

        Args:
            memory_store_provider: Optional function that returns a memory store.
                                  If not provided, uses the admitted store
                                  published by the lifecycle manager.
        """
        self.get_memory_store = (
            memory_store_provider
            if memory_store_provider is not None
            else get_memory_store
        )

        self.router = APIRouter(prefix="/api/memory", tags=["memory"])
        self._setup_routes()

    @property
    def memory_store(self) -> MemoryStore:
        """Get the published memory store, or fail closed with a 503.

        Every lifecycle condition the runtime can be in -- awaiting a retryable
        admission, blocked on repair, missing a usable credential, or needing a
        restart after meaningful drift -- reaches callers as the same stable,
        public-safe 503. Only the operator log carries the detail.
        """
        try:
            return self.get_memory_store()
        except MemoryUnavailableError as error:
            raise HTTPException(status_code=503, detail=error.status.detail) from None

    def _setup_routes(self) -> None:
        @self.router.get("/list", response_model=MemoryListResponse)
        async def list_memories(
            category: Optional[str] = Query(None, description="Filter by category"),
            tags: Optional[str] = Query(
                None, description="Comma-separated tags to filter"
            ),
            keywords: Optional[str] = Query(
                None, description="Comma-separated keywords to filter"
            ),
            date_from: Optional[datetime] = Query(
                None, description="Filter from this date"
            ),
            date_to: Optional[datetime] = Query(
                None, description="Filter to this date"
            ),
            search: Optional[str] = Query(
                None, description="Search query to filter memories by content"
            ),
            space: Optional[Literal["personal", "company", "department"]] = Query(None),
            scope: Optional[Literal["owner", "visible", "org"]] = Query(None),
            space_id: Optional[int] = Query(None, ge=1),
            similarity_threshold: Optional[float] = Query(
                None,
                ge=0.1,
                le=2.0,
                description="Similarity threshold for vector search (0.1-2.0)",
            ),
            limit: int = Query(
                100, ge=1, le=1000, description="Maximum results to return"
            ),
            offset: int = Query(0, ge=0, description="Offset for pagination"),
            user: User = Depends(get_current_user),
        ) -> MemoryListResponse:
            try:
                # Set user context for memory operations
                with UserContext(int(user.id)):
                    # Build filters
                    filters: dict[str, Any] = {}
                    store = self.memory_store
                    if space is not None or scope is not None or space_id is not None:
                        if not isinstance(
                            unwrap_memory_store(store), RemoteMemoryStore
                        ):
                            raise HTTPException(
                                400,
                                "Space/scope selection is unsupported by this memory backend",
                            )
                    if space_id is not None:
                        raise HTTPException(
                            400,
                            "space_id filtering is not supported by the current memory service contract",
                        )
                    if space is not None:
                        filters["space"] = space
                    if category:
                        filters["category"] = category
                    if tags:
                        filters["tags"] = [
                            tag.strip() for tag in tags.split(",") if tag.strip()
                        ]
                    if keywords:
                        filters["keywords"] = [
                            kw.strip() for kw in keywords.split(",") if kw.strip()
                        ]
                    if date_from:
                        filters["date_from"] = date_from
                    if date_to:
                        filters["date_to"] = date_to

                    transport_filters = dict(filters)
                    if scope is not None:
                        transport_filters[READ_SCOPE_FILTER_KEY] = scope

                    try:
                        with memory_operation():
                            if search:
                                # Embeds the query.
                                async with usage_scope(
                                    int(user.id), usage_source="memory_api"
                                ):
                                    memories = store.search(
                                        query=search,
                                        k=1000,
                                        filters=transport_filters or None,
                                        similarity_threshold=similarity_threshold,
                                    )
                            else:
                                memories = store.list_all(transport_filters)
                    except HTTPException:
                        raise
                    except Exception:
                        logger.exception("Memory list read failed")
                        raise HTTPException(
                            status_code=503,
                            detail=MEMORY_READ_UNAVAILABLE_DETAIL,
                        ) from None

                    # Apply pagination
                    total_count = len(memories)
                    memories = memories[offset : offset + limit]

                    # Convert to dict format for response
                    memory_dicts = []
                    for memory in memories:
                        memory_dict: dict[str, Any] = {
                            "id": memory.id,
                            "content": memory.content,
                            "keywords": memory.keywords,
                            "tags": memory.tags,
                            "category": memory.category,
                            "timestamp": memory.timestamp,
                            "mime_type": memory.mime_type,
                            "metadata": memory.metadata,
                        }
                        if isinstance(memory, RemoteMemoryNote):
                            memory_dict.update(
                                space_kind=memory.space_kind, space_id=memory.space_id
                            )
                        memory_dicts.append(memory_dict)

                    return MemoryListResponse(
                        memories=memory_dicts,
                        total_count=total_count,
                        filters_used={**filters, **({"scope": scope} if scope else {})},
                    )
            except HTTPException:
                raise
            except Exception:
                logger.exception("Failed to build memory list response")
                raise HTTPException(
                    status_code=500, detail="Failed to list memories."
                ) from None

        @self.router.delete(MEMORY_ID_ROUTE)
        async def delete_memory(
            memory_id: MemoryIdPath, user: User = Depends(get_current_user)
        ) -> dict[str, Any]:
            try:
                # Set user context for memory operations
                with UserContext(int(user.id)), memory_operation():
                    response = self.memory_store.delete(memory_id)
                    if response.success:
                        return {
                            "success": True,
                            "message": "Memory deleted successfully",
                        }
                    else:
                        raise HTTPException(
                            status_code=404, detail=MEMORY_NOT_FOUND_DETAIL
                        )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to delete memory: {str(e)}"
                )

        @self.router.put(MEMORY_ID_ROUTE)
        async def update_memory(
            memory_id: MemoryIdPath,
            update_request: MemoryUpdateRequest,
            user: User = Depends(get_current_user),
        ) -> dict[str, Any]:
            if {"space", "space_kind", "space_id"} & update_request.model_fields_set:
                raise HTTPException(
                    400, "Changing a memory's space is not supported by ordinary edits"
                )
            try:
                # Set user context for memory operations
                with UserContext(int(user.id)), memory_operation():
                    # Get existing memory
                    get_response = self.memory_store.get(memory_id)
                    if not get_response.success:
                        raise HTTPException(
                            status_code=404, detail=MEMORY_NOT_FOUND_DETAIL
                        )

                    existing_memory = get_response.content
                    if not isinstance(existing_memory, MemoryNote):
                        raise HTTPException(
                            status_code=500, detail="Invalid memory data"
                        )

                    # Update fields that are provided
                    updates: dict[str, Any] = {}
                    if update_request.content is not None:
                        updates["content"] = update_request.content
                    if update_request.keywords is not None:
                        updates["keywords"] = update_request.keywords
                    if update_request.tags is not None:
                        updates["tags"] = update_request.tags
                    if update_request.category is not None:
                        updates["category"] = update_request.category
                    if update_request.metadata is not None:
                        updates["metadata"] = update_request.metadata

                    # Create updated memory note
                    updated_memory = MemoryNote(
                        id=memory_id,
                        content=updates.get("content", existing_memory.content),
                        keywords=updates.get("keywords", existing_memory.keywords),
                        tags=updates.get("tags", existing_memory.tags),
                        category=updates.get("category", existing_memory.category),
                        metadata=updates.get("metadata", existing_memory.metadata),
                        mime_type=existing_memory.mime_type,
                        timestamp=existing_memory.timestamp,  # Keep original timestamp
                    )

                    # Update in store; re-embeds the note.
                    async with usage_scope(int(user.id), usage_source="memory_api"):
                        response = self.memory_store.update(updated_memory)
                    if response.success:
                        return {
                            "success": True,
                            "message": "Memory updated successfully",
                        }
                    else:
                        raise HTTPException(
                            status_code=500,
                            detail=response.error or "Failed to update memory",
                        )

            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to update memory: {str(e)}"
                )

        @self.router.get("/stats", response_model=MemoryStatsResponse)
        async def get_memory_stats(
            user: User = Depends(get_current_user),
        ) -> MemoryStatsResponse:
            try:
                # Set user context for memory operations
                with UserContext(int(user.id)), memory_operation():
                    stats = self.memory_store.get_stats()
                    return MemoryStatsResponse(**stats)
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to get memory stats: {str(e)}"
                )

        @self.router.post("")
        async def create_memory(
            memory_request: dict[str, Any], user: User = Depends(get_current_user)
        ) -> dict[str, Any]:
            # Validate required fields
            content = memory_request.get("content")
            if not isinstance(content, str) or not content.strip():
                raise HTTPException(
                    status_code=422, detail="Non-blank content is required"
                )
            # Resolved before the try block on purpose: the lifecycle 503 must
            # reach the caller as itself, while the handler below keeps
            # wrapping store failures the way it always has.
            store = self.memory_store
            remote = isinstance(unwrap_memory_store(store), RemoteMemoryStore)
            if (
                not remote
                and {"space", "space_kind", "space_id"} & memory_request.keys()
            ):
                raise HTTPException(
                    400, "Spaces are unsupported by this memory backend"
                )
            space = memory_request.get("space", "personal")
            if not isinstance(space, str) or space not in {
                "personal",
                "company",
                "department",
            }:
                raise HTTPException(400, "Unsupported memory space")
            kind = None if space == "personal" else space
            if "space_kind" in memory_request:
                supplied = memory_request["space_kind"]
                if supplied is not None and not isinstance(supplied, str):
                    raise HTTPException(400, "Unsupported space_kind")
                if "space" in memory_request and supplied != kind:
                    raise HTTPException(400, "Conflicting space and space_kind")
                if supplied not in {None, "company", "department"}:
                    raise HTTPException(400, "Unsupported space_kind")
                kind = supplied
            space_id = memory_request.get("space_id")
            if kind == "department":
                if type(space_id) is not int or space_id <= 0:
                    raise HTTPException(
                        400, "A department space requires a positive integer space_id"
                    )
            elif space_id is not None:
                raise HTTPException(
                    400, "space_id is only valid for a department space"
                )
            try:
                # Set user context for memory operations
                with UserContext(int(user.id)), memory_operation():
                    # Create new memory note
                    note_type = RemoteMemoryNote if remote else MemoryNote
                    memory_note = note_type(
                        content=memory_request.get("content", ""),
                        keywords=memory_request.get("keywords", []),
                        tags=memory_request.get("tags", []),
                        category=memory_request.get("category", "general"),
                        metadata=memory_request.get("metadata", {}),
                        **(
                            {"space_kind": kind, "space_id": space_id} if remote else {}
                        ),
                    )

                    # Embeds the note.
                    async with usage_scope(int(user.id), usage_source="memory_api"):
                        response = store.add(memory_note)
                    if response.success:
                        return {
                            "success": True,
                            "memory_id": response.memory_id,
                            "message": "Memory created successfully",
                            **(
                                {"space_kind": kind, "space_id": space_id}
                                if remote
                                else {}
                            ),
                        }
                    else:
                        raise HTTPException(
                            status_code=500,
                            detail=response.error or "Failed to create memory",
                        )

            except HTTPException:
                # Needed for the 503 above to reach the caller at all: without
                # it the broad handler re-wrapped every HTTPException raised
                # inside the block, status code included, as a 500.
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to create memory: {str(e)}"
                )

        # Registered before the "/{memory_id}" catch-all, which would otherwise
        # swallow this path and answer it as a lookup for a note called
        # "store-info".
        @self.router.get("/store-info")
        async def get_store_info(user: User = Depends(get_current_user)) -> dict:
            """Report the memory lifecycle state.

            This stays available in every lifecycle state, including
            BLOCKED_REPAIR, so an operator can see why memory is fenced off
            while the rest of the deployment keeps serving.
            """
            try:
                # Off the event loop, like every other caller of the manager:
                # the report reaches the drift check, which checks out a
                # synchronous authority Session. Running that inline would
                # block the loop thread while ``get_current_user``'s
                # request-scoped connection is still held, so on a
                # single-slot pool the nested checkout would be waiting on
                # the very thread that has to release it.
                with UserContext(int(user.id)):
                    return await run_db_io_cancellation_safe(
                        get_memory_store_manager().get_store_info
                    )
            except RemoteMemoryStoreError:
                raise HTTPException(
                    status_code=503, detail=MEMORY_READ_UNAVAILABLE_DETAIL
                ) from None
            except Exception:
                logger.exception("Failed to read memory store info")
                raise HTTPException(
                    status_code=500, detail="Failed to get store info."
                ) from None

        @self.router.get(MEMORY_ID_ROUTE)
        async def get_memory(
            memory_id: MemoryIdPath, user: User = Depends(get_current_user)
        ) -> dict[str, Any]:
            try:
                # Set user context for memory operations
                with UserContext(int(user.id)), memory_operation():
                    response = self.memory_store.get(memory_id)
                    if response.success and response.content:
                        memory = response.content
                        if isinstance(memory, MemoryNote):
                            return {
                                "id": memory.id,
                                "content": memory.content,
                                "keywords": memory.keywords,
                                "tags": memory.tags,
                                "category": memory.category,
                                "timestamp": memory.timestamp,
                                "mime_type": memory.mime_type,
                                "metadata": memory.metadata,
                                **(
                                    {
                                        "space_kind": memory.space_kind,
                                        "space_id": memory.space_id,
                                    }
                                    if isinstance(memory, RemoteMemoryNote)
                                    else {}
                                ),
                            }
                        else:
                            return {"content": memory}
                    else:
                        raise HTTPException(
                            status_code=404, detail=MEMORY_NOT_FOUND_DETAIL
                        )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to get memory: {str(e)}"
                )

    def get_router(self) -> APIRouter:
        return self.router
