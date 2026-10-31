"""Synthetic regressions for memory API validation and configuration."""

from datetime import datetime
from importlib import import_module
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from xagent.core.memory.core import MemoryNote
from xagent.core.memory.in_memory import InMemoryMemoryStore
from xagent.web.api.memory import MemoryManagementRouter
from xagent.web.auth_dependencies import get_current_user
from xagent.web.memory_lifecycle import default_similarity_threshold
from xagent.web.models.user import User
from xagent.web.schemas.chat import TaskCreateResponse
from xagent.web.user_isolated_memory import UserContext, UserIsolatedMemoryStore


@pytest.fixture
def memory_client():
    store = UserIsolatedMemoryStore(InMemoryMemoryStore())
    identity = {"user_id": 31}
    app = FastAPI()
    app.include_router(MemoryManagementRouter(lambda: store).get_router())
    app.dependency_overrides[get_current_user] = lambda: User(
        id=identity["user_id"], username="synthetic", password_hash="unused"
    )
    with TestClient(app) as client:
        yield client, store, identity


@pytest.mark.parametrize("method", ["GET", "PUT", "DELETE"])
@pytest.mark.parametrize("target", ["missing", "another-owner"])
def test_missing_and_hidden_notes_have_one_public_404(memory_client, method, target):
    client, store, _identity = memory_client
    with UserContext(32):
        assert store.add(
            MemoryNote(id="another-owner", content="synthetic secret")
        ).success
    response = client.request(
        method,
        f"/api/memory/{target}",
        json={"content": "synthetic edit"} if method == "PUT" else None,
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "not found or not accessible"}
    with UserContext(32):
        assert store.get("another-owner").content.content == "synthetic secret"


@pytest.mark.parametrize("content", ["", "  ", "\t\n\u3000"])
def test_blank_create_and_edit_are_rejected_without_mutating(memory_client, content):
    client, store, _identity = memory_client
    with UserContext(31):
        assert store.add(
            MemoryNote(id="existing", content="keep synthetic value")
        ).success
    assert client.post("/api/memory", json={"content": content}).status_code == 422
    assert (
        client.put("/api/memory/existing", json={"content": content}).status_code == 422
    )
    with UserContext(31):
        assert store.get("existing").content.content == "keep synthetic value"
        assert len(store.list_all()) == 1


def test_omitting_content_preserves_it_during_an_edit(memory_client):
    client, store, _identity = memory_client
    with UserContext(31):
        assert store.add(
            MemoryNote(id="existing", content="keep synthetic value")
        ).success
    response = client.put("/api/memory/existing", json={"tags": ["synthetic"]})
    assert response.status_code == 200
    with UserContext(31):
        note = store.get("existing").content
    assert note.content == "keep synthetic value"
    assert note.tags == ["synthetic"]


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, 0.4),
        ("invalid", 0.4),
        ("nan", 0.4),
        ("inf", 0.4),
        ("-1", 0.1),
        ("3", 2.0),
        ("0.25", 0.25),
    ],
)
def test_default_threshold_is_finite_and_bounded(monkeypatch, configured, expected):
    monkeypatch.delenv("MEMORY_SIMILARITY_THRESHOLD", raising=False)
    if configured is not None:
        monkeypatch.setenv("MEMORY_SIMILARITY_THRESHOLD", configured)
    assert default_similarity_threshold() == expected


def test_task_create_explains_that_the_client_must_dispatch():
    result = TaskCreateResponse(
        task_id=1,
        title="synthetic",
        status="pending",
        created_at=datetime.now().isoformat(),
        connector_runtime_requirements=None,
    )
    assert result.model_dump()["dispatch_required"] is True


def test_readiness_alias_preserves_startup_failure_instead_of_claiming_health(
    monkeypatch,
):
    module = import_module("xagent.web.app")
    routes = {getattr(route, "path", ""): route for route in module.app.routes}
    assert routes["/readyz"].endpoint is routes["/ready"].endpoint
    assert routes["/healthz"].endpoint is routes["/health"].endpoint
    monkeypatch.setattr(
        module.app.state,
        "file_storage_startup_sync_task",
        SimpleNamespace(done=lambda: False),
        raising=False,
    )
    monkeypatch.setattr(
        module.app.state, "file_storage_startup_sync_error", None, raising=False
    )
    isolated = FastAPI()
    for path in ("/ready", "/readyz", "/healthz"):
        isolated.add_api_route(path, routes[path].endpoint)
    with TestClient(isolated) as client:
        assert client.get("/ready").status_code == 503
        assert client.get("/readyz").status_code == 503
        assert client.get("/healthz").status_code == 200
