"""Remote adapter contracts through the authenticated memory HTTP surface.

The session is entirely in-process: these tests exercise the actual adapter,
router, and response envelopes without calling a memory service or model.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast
from urllib.parse import quote

import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

from xagent.core.memory.base import MemoryStore
from xagent.core.memory.in_memory import InMemoryMemoryStore
from xagent.core.memory.remote import RemoteMemoryStore, RemoteMemoryStoreIdentityError
from xagent.core.user_context import current_user_id
from xagent.web.api import memory as memory_api
from xagent.web.auth_dependencies import get_current_user
from xagent.web.models.user import User

USER_ID = 17
NOTE_ID = 'opaque /%_" note'


class ScriptedSession:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.status = 200
        self.payload: Any = {}
        self.failure: requests.exceptions.RequestException | None = None

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        self.calls.append({"method": method, "url": url, **kwargs})
        assert kwargs["headers"]["X-User-External-Id"] == str(USER_ID)
        if self.failure is not None:
            raise self.failure
        response = requests.Response()
        # The editor first reads the note. Its subsequent mutation receives
        # the scripted receipt/error, rather than a fabricated complete note.
        if method == "GET" and "/notes/" in url:
            response.status_code = 200
            payload: Any = {"id": NOTE_ID, "content": "before"}
        else:
            response.status_code = self.status
            payload = self.payload
        response._content = json.dumps(payload).encode()
        return response


def make_client(store: MemoryStore) -> TestClient:
    app = FastAPI()
    app.include_router(memory_api.MemoryManagementRouter(lambda: store).get_router())
    app.dependency_overrides[get_current_user] = lambda: User(
        id=USER_ID, username="memory-tests", password_hash="unused"
    )
    return TestClient(app)


@pytest.fixture
def remote_client() -> Iterator[tuple[TestClient, ScriptedSession]]:
    session = ScriptedSession()
    adapter = RemoteMemoryStore(
        "http://memsrv.invalid:8080",
        "test-runtime-key",
        session=cast(requests.Session, session),
    )
    with make_client(adapter) as client:
        yield client, session


@pytest.mark.parametrize("truncated", [False, True, None, "false", 0])
def test_remote_list_completeness_precedes_web_total(
    remote_client: tuple[TestClient, ScriptedSession], truncated: Any
) -> None:
    client, session = remote_client
    session.payload = {
        "results": [
            {"id": "one", "content": "first"},
            {"id": "two", "content": "second"},
        ],
        "limit": 10000,
        "scope": "personal",
    }
    if truncated is not None:
        session.payload["truncated"] = truncated

    response = client.get("/api/memory/list?category=general&limit=1")

    assert len(session.calls) == 1
    assert session.calls[0]["json"] == {"filters": {"category": "general"}}
    if truncated is False:
        assert response.status_code == 200
        assert response.json()["total_count"] == 2
        assert [note["id"] for note in response.json()["memories"]] == ["one"]
    else:
        assert response.status_code == 503
        assert response.json() == {"detail": memory_api.MEMORY_READ_UNAVAILABLE_DETAIL}


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE"])
def test_remote_mutation_conflict_is_not_reported_as_success(
    remote_client: tuple[TestClient, ScriptedSession], method: str
) -> None:
    client, session = remote_client
    session.status = 409
    session.payload = {
        "error": {"code": "conflict", "message": "delivered notes are append-only"}
    }
    path = (
        "/api/memory" if method == "POST" else f"/api/memory/{quote(NOTE_ID, safe='')}"
    )
    kwargs: dict[str, Any] = (
        {"json": {"content": "after"}} if method != "DELETE" else {}
    )

    response = client.request(method, path, **kwargs)

    assert response.status_code == 409
    assert "append-only" in response.json()["detail"]
    assert "success" not in response.json()
    assert session.calls[-1]["method"] == ("PATCH" if method == "PUT" else method)


@pytest.mark.parametrize("path", ["/api/memory/list", "/api/memory/stats"])
def test_remote_dependency_failure_is_public_safe_unavailable(
    remote_client: tuple[TestClient, ScriptedSession], path: str
) -> None:
    client, session = remote_client
    session.status = 503
    session.payload = {"error": {"message": "private upstream diagnostic"}}

    response = client.get(path)

    assert response.status_code == 503
    assert response.json() == {"detail": memory_api.MEMORY_READ_UNAVAILABLE_DETAIL}


def test_remote_patch_accepts_receipt_and_preserves_opaque_id(
    remote_client: tuple[TestClient, ScriptedSession],
) -> None:
    client, session = remote_client
    session.payload = {"id": NOTE_ID, "success": True, "owner_id": 24}

    response = client.put(
        f"/api/memory/{quote(NOTE_ID, safe='')}", json={"content": "after", "tags": []}
    )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert [call["method"] for call in session.calls] == ["GET", "PATCH"]
    assert all(call["url"].endswith(quote(NOTE_ID, safe="")) for call in session.calls)
    assert session.calls[-1]["json"]["content"] == "after"
    assert session.calls[-1]["json"]["tags"] == []
    assert (
        not {"id", "parent_id", "owner_id", "space_id"}
        & session.calls[-1]["json"].keys()
    )


@pytest.mark.parametrize("status", [404, 405])
def test_remote_patch_never_falls_back_to_put(
    remote_client: tuple[TestClient, ScriptedSession], status: int
) -> None:
    client, session = remote_client
    session.status = status
    session.payload = {"error": {"message": "patch unavailable"}}

    response = client.put(
        f"/api/memory/{quote(NOTE_ID, safe='')}", json={"content": "after"}
    )

    assert response.status_code >= 400
    assert "success" not in response.json()
    assert [call["method"] for call in session.calls] == ["GET", "PATCH"]


def test_store_info_enters_trusted_user_context_before_worker_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = []

    def report() -> dict[str, Any]:
        seen.append(current_user_id.get())
        return {"state": "degraded", "supports_vector_search": False}

    monkeypatch.setattr(
        memory_api,
        "get_memory_store_manager",
        lambda: SimpleNamespace(get_store_info=report),
    )
    with make_client(InMemoryMemoryStore()) as client:
        response = client.get("/api/memory/store-info")

    assert response.status_code == 200
    assert response.json() == {"state": "degraded", "supports_vector_search": False}
    assert seen == [USER_ID]


def test_store_info_does_not_expose_failed_identity_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def report() -> dict[str, Any]:
        assert current_user_id.get() == USER_ID
        raise RemoteMemoryStoreIdentityError("private mapping/configuration diagnostic")

    monkeypatch.setattr(
        memory_api,
        "get_memory_store_manager",
        lambda: SimpleNamespace(get_store_info=report),
    )
    with make_client(InMemoryMemoryStore()) as client:
        response = client.get("/api/memory/store-info")

    assert response.status_code == 503
    assert response.json() == {"detail": memory_api.MEMORY_READ_UNAVAILABLE_DETAIL}


def test_local_memory_crud_remains_available() -> None:
    with make_client(InMemoryMemoryStore()) as client:
        created = client.post("/api/memory", json={"content": "local"})
        assert created.status_code == 200
        note_id = created.json()["memory_id"]
        assert client.get(f"/api/memory/{note_id}").json()["content"] == "local"
        assert (
            client.put(
                f"/api/memory/{note_id}", json={"content": "changed"}
            ).status_code
            == 200
        )
        assert client.delete(f"/api/memory/{note_id}").status_code == 200
        assert client.get(f"/api/memory/{note_id}").status_code == 404
