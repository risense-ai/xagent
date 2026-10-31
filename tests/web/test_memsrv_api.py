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
        self.note_payload: Any = {"id": NOTE_ID, "content": "before"}
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
            payload: Any = self.note_payload
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


def test_space_and_read_scope_use_separate_wire_fields(remote_client):
    client, session = remote_client
    session.payload = {
        "results": [
            {
                "id": "shared",
                "content": "synthetic",
                "space_kind": "company",
                "space_id": None,
            }
        ],
        "truncated": False,
    }
    response = client.get("/api/memory/list?space=company&scope=visible")
    assert response.status_code == 200
    assert session.calls[-1]["json"] == {
        "scope": "visible",
        "filters": {"space": "company"},
    }
    assert response.json()["filters_used"] == {"space": "company", "scope": "visible"}
    assert response.json()["memories"][0]["space_kind"] == "company"


@pytest.mark.parametrize("scope", ["owner", "org"])
def test_search_rejects_unsupported_scope_without_claiming_it_was_applied(
    remote_client, scope
):
    client, session = remote_client
    response = client.get(
        f"/api/memory/list?search=synthetic&scope={scope}&space=company"
    )
    assert response.status_code == 400
    assert "scope" in response.json()["detail"]
    assert "filters_used" not in response.json()
    assert session.calls == []


def test_search_visible_scope_and_space_have_supported_semantics(remote_client):
    client, session = remote_client
    session.payload = {"results": []}
    response = client.get(
        "/api/memory/list?search=synthetic&scope=visible&space=company"
    )
    assert response.status_code == 200
    assert response.json()["filters_used"] == {"space": "company", "scope": "visible"}
    assert session.calls[-1]["json"]["scope"] == "visible"
    assert session.calls[-1]["json"]["filters"] == {"space": "company"}


@pytest.mark.parametrize("scope", ["owner", "org"])
def test_listing_still_sends_the_supported_scope_selection(remote_client, scope):
    client, session = remote_client
    session.payload = {"results": [], "truncated": False}
    response = client.get(f"/api/memory/list?scope={scope}&space=company")
    assert response.status_code == 200
    assert response.json()["filters_used"] == {"scope": scope, "space": "company"}
    assert session.calls[-1]["json"] == {
        "scope": scope,
        "filters": {"space": "company"},
    }


def test_listing_encodes_dates_and_paginates_a_complete_scoped_result(remote_client):
    client, session = remote_client
    session.payload = {
        "results": [
            {
                "id": str(index),
                "content": "synthetic",
                "timestamp": f"2026-10-31T{hour}:00:00Z",
            }
            for index, hour in enumerate(("11", "10", "09"))
        ],
        "truncated": False,
    }
    response = client.get(
        "/api/memory/list",
        params={
            "scope": "owner",
            "space": "personal",
            "date_from": "2026-10-30T00:00:00Z",
            "date_to": "2026-11-01T00:00:00+08:00",
            "limit": 1,
            "offset": 1,
        },
    )
    assert response.status_code == 200
    assert response.json()["total_count"] == 3
    assert [note["id"] for note in response.json()["memories"]] == ["1"]
    assert session.calls[-1]["json"] == {
        "scope": "owner",
        "filters": {
            "space": "personal",
            "date_from": "2026-10-30T00:00:00+00:00",
            "date_to": "2026-10-31T16:00:00+00:00",
        },
    }


def test_workspace_is_not_silently_treated_as_a_memory_space(remote_client):
    client, session = remote_client
    assert client.get("/api/memory/list?space=workspace").status_code == 422
    assert session.calls == []


def test_explicit_space_write_and_get_preserve_literal_content(remote_client):
    client, session = remote_client
    session.payload = {"id": NOTE_ID, "success": True}
    text = "[share:company] is literal synthetic text"
    created = client.post(
        "/api/memory", json={"content": text, "space": "department", "space_id": 7}
    )
    assert created.status_code == 200
    wire = session.calls[-1]
    assert wire["url"].endswith("/v1/memory/notes")
    assert wire["json"]["content"] == text
    assert wire["json"]["space_kind"] == "department"
    assert wire["json"]["space_id"] == 7
    assert created.json()["space_kind"] == "department"
    session.note_payload = {
        "id": NOTE_ID,
        "content": text,
        "space_kind": "department",
        "space_id": 7,
    }
    fetched = client.get(f"/api/memory/{quote(NOTE_ID, safe='')}")
    assert fetched.status_code == 200
    assert fetched.json()["space_id"] == 7
    assert fetched.json()["content"] == text


@pytest.mark.parametrize(
    "body",
    [
        {"space": "company", "space_kind": "department", "space_id": 1},
        {"space": "department"},
        {"space": "personal", "space_id": 1},
        {"space": "department", "space_id": True},
        {"space": "unsupported"},
    ],
)
def test_invalid_or_conflicting_spaces_fail_before_http(remote_client, body):
    client, session = remote_client
    assert (
        client.post("/api/memory", json={"content": "synthetic", **body}).status_code
        == 400
    )
    assert session.calls == []


def test_unsupported_space_id_filter_is_explicit(remote_client):
    client, session = remote_client
    assert client.get("/api/memory/list?space=department&space_id=7").status_code == 400
    assert session.calls == []


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


@pytest.mark.parametrize("status", [403, 404])
def test_edit_denied_after_successful_read_has_the_same_public_404(
    remote_client, status
):
    client, session = remote_client
    session.status = status
    session.payload = {"error": {"message": "SYNTHETIC_PRIVATE_ACCESS_DIAGNOSTIC"}}
    response = client.put(
        f"/api/memory/{quote(NOTE_ID, safe='')}", json={"content": "synthetic edit"}
    )
    assert [call["method"] for call in session.calls] == ["GET", "PATCH"]
    assert response.status_code == 404
    assert response.json() == {"detail": "not found or not accessible"}


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


@pytest.mark.parametrize("search", [None, "synthetic query"])
def test_web_date_filters_use_real_requests_json_encoding(search: str | None) -> None:
    from datetime import datetime

    class WireSession(requests.Session):
        def __init__(self) -> None:
            super().__init__()
            self.sent: list[requests.PreparedRequest] = []

        def send(
            self, request: requests.PreparedRequest, **kwargs: Any
        ) -> requests.Response:
            self.sent.append(request)
            response = requests.Response()
            response.status_code = 200
            response._content = b'{"results":[],"truncated":false}'
            return response

    session = WireSession()
    store = RemoteMemoryStore("http://memsrv.invalid", "fixture", session=session)
    params = {
        "date_from": "2026-10-30T09:15:00+08:00",
        "date_to": "2026-10-31T00:00:00Z",
    }
    if search:
        params["search"] = search
    with make_client(store) as client:
        response = client.get("/api/memory/list", params=params)
    assert response.status_code == 200
    assert response.json()["total_count"] == 0
    assert len(session.sent) == 1
    assert session.sent[0].headers["X-User-External-Id"] == str(USER_ID)
    filters = json.loads(session.sent[0].body)["filters"]
    for key in ("date_from", "date_to"):
        assert (
            datetime.fromisoformat(filters[key]).timestamp()
            == datetime.fromisoformat(params[key]).timestamp()
        )
