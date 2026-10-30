"""Real adapter/native wrapper contracts; all HTTP answers are synthetic."""

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import Mock

import pytest
import requests

from xagent.core.execution_scope import (
    ExecutionScope,
    reset_execution_scope,
    set_execution_scope,
)
from xagent.core.memory.access_scope import MemoryAccessScope, bind_memory_access_scope
from xagent.core.memory.core import MemoryNote
from xagent.core.memory.remote import (
    RemoteMemoryStore,
    RemoteMemoryStoreError,
    RemoteMemoryStoreIdentityError,
    RemoteMemoryStoreIncompleteError,
)
from xagent.core.user_context import current_user_id
from xagent.web.revocable_memory_store import RevocableMemoryStore

DIM = "execution_scope_client_application_id"
OTHER_DIM = "execution_scope_connector_id"


def reply(payload: Any, status: int = 200) -> requests.Response:
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(payload).encode()
    return result


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_URL", "http://memory-internal")
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_KEY", "fixture-token")
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "false")
    monkeypatch.delenv("XAGENT_MEMORY_AGENT_BINDINGS", raising=False)
    session = Mock(spec=requests.Session)
    raw = RemoteMemoryStore("http://memory-internal", "fixture-token", session=session)
    bound = bind_memory_access_scope(
        raw, MemoryAccessScope(31, {"client_application_id": "42"})
    )
    token = current_user_id.set(31)
    try:
        yield raw, bound, session
    finally:
        current_user_id.reset(token)


def test_fixed_dimensions_override_ambient_and_forged_dimensions(backend):
    raw, bound, session = backend
    assert bound is not raw
    token = set_execution_scope(
        ExecutionScope(memory_dimensions={"connector_id": "ambient"})
    )
    try:
        session.request.return_value = reply({"id": "created"})
        note = MemoryNote(
            content="synthetic",
            metadata={DIM: "forged", OTHER_DIM: "extra", "user_id": 999},
        )
        assert bound.add(note).success
        payload = session.request.call_args.kwargs["json"]
        assert payload["metadata"] == {DIM: "42"}
        # The platform User.id selects the server identity; it is not the
        # physical memsrv person/seat ID in returned metadata.
        assert session.request.call_args.kwargs["headers"]["X-User-External-Id"] == "31"
        session.request.return_value = reply({"results": [], "truncated": False})
        bound.search(
            "query",
            filters={"metadata": {DIM: "forged", OTHER_DIM: "extra", "business": "ok"}},
        )
        assert session.request.call_args.kwargs["json"]["filters"] == {
            "metadata": {DIM: "42", "business": "ok"}
        }
        bound.list_all()
        assert session.request.call_args.kwargs["json"]["filters"] == {
            "metadata": {DIM: "42"}
        }
    finally:
        reset_execution_scope(token)


@pytest.mark.parametrize("operation", ["get", "update", "delete"])
def test_fixed_by_id_denies_other_namespace_before_write(backend, operation):
    _, bound, session = backend
    session.request.return_value = reply(
        {
            "id": "opaque-id",
            "content": "other",
            "metadata": {DIM: "other", "user_id": 202},
        }
    )
    argument = (
        MemoryNote(id="opaque-id", content="edit")
        if operation == "update"
        else "opaque-id"
    )
    assert not getattr(bound, operation)(argument).success
    assert session.request.call_count == 1
    assert session.request.call_args.args[0] == "GET"


def test_fixed_patch_preserves_namespace_and_only_editable_fields(backend):
    _, bound, session = backend
    session.request.side_effect = [
        reply(
            {"id": "note", "content": "old", "metadata": {DIM: "42", "user_id": 202}}
        ),
        reply({"id": "note", "success": True}),
    ]
    assert bound.update(
        MemoryNote(
            id="note",
            content="new",
            metadata={DIM: "forged", OTHER_DIM: "extra", "user_id": 202},
        )
    ).success
    call = session.request.call_args
    assert call.args[0] == "PATCH"
    assert call.kwargs["json"] == {
        "content": "new",
        "metadata": {DIM: "42", "user_id": 202},
    }


@pytest.mark.parametrize("operation", ["clear", "get_stats"])
def test_fixed_bulk_never_uses_user_wide_endpoint(backend, operation):
    _, bound, session = backend
    note = {
        "id": "note",
        "content": "body",
        "metadata": {DIM: "42", "user_id": 202},
        "category": "facts",
        "tags": ["t"],
    }
    session.request.side_effect = [
        reply({"results": [note], "truncated": False}),
        reply(note),
        reply({"success": True}),
    ]
    result = getattr(bound, operation)()
    paths = [call.args[1] for call in session.request.call_args_list]
    assert not any(path.endswith(("/clear", "/stats")) for path in paths)
    if operation == "get_stats":
        assert result == {
            "total_count": 1,
            "category_counts": {"facts": 1},
            "tag_counts": {"t": 1},
            "memory_store_type": "remote",
        }
    else:
        assert session.request.call_args.args[0] == "DELETE"


def test_truncated_fixed_clear_does_not_delete_anything(backend):
    _, bound, session = backend
    session.request.return_value = reply(
        {"results": [{"id": "note"}], "truncated": True}
    )
    with pytest.raises(RemoteMemoryStoreIncompleteError):
        bound.clear()
    assert session.request.call_count == 1
    assert session.request.call_args.args[0] == "POST"
    assert session.request.call_args.args[1].endswith("/list")


@pytest.mark.parametrize(
    "change", ["missing", "zero", "blank", "user", "url", "key", "mode", "mapping"]
)
def test_observed_native_scope_failure_cannot_revive_after_restore(
    backend, monkeypatch, change
):
    _, bound, session = backend
    session.request.return_value = reply({"results": [], "truncated": False})
    bound.search("before")
    if change in {"missing", "zero", "blank", "user"}:
        current_user_id.set(
            {"missing": None, "zero": 0, "blank": " ", "user": 32}[change]
        )
    else:
        name = {
            "url": "SERVICE_URL",
            "key": "SERVICE_KEY",
            "mode": "AGENT_SEATS",
            "mapping": "AGENT_BINDINGS",
        }[change]
        monkeypatch.setenv(
            f"XAGENT_MEMORY_{name}", "true" if change == "mode" else "changed"
        )
    with pytest.raises(RemoteMemoryStoreIdentityError):
        bound.search("changed")
    current_user_id.set(31)
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_URL", "http://memory-internal")
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_KEY", "fixture-token")
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "false")
    monkeypatch.delenv("XAGENT_MEMORY_AGENT_BINDINGS", raising=False)
    with pytest.raises(RemoteMemoryStoreIdentityError):
        bound.search("restored")
    assert session.request.call_count == 1


def test_cross_owner_capability_fails_closed_without_selecting_ambient_user(backend):
    raw, _, session = backend
    other = bind_memory_access_scope(
        raw, MemoryAccessScope(32, {"client_application_id": "42"})
    )
    with pytest.raises(RemoteMemoryStoreIdentityError):
        other.search("must not select user31")
    session.request.assert_not_called()


def test_fresh_rebind_and_worker_context_preserve_capability(backend):
    raw, first, session = backend
    rebound = bind_memory_access_scope(
        first, MemoryAccessScope(31, {"client_application_id": "different"})
    )
    session.request.return_value = reply({"results": [], "truncated": False})

    def worker():
        token = current_user_id.set(31)
        try:
            rebound.search("thread")
        finally:
            current_user_id.reset(token)

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(worker).result()
    assert session.request.call_args.kwargs["json"]["filters"] == {
        "metadata": {DIM: "different"}
    }
    assert rebound is not first and rebound is not raw


@pytest.mark.parametrize(
    "operation",
    [
        "remember",
        "inbox_list",
        "inbox_read",
        "delete_by_scope_dimension",
        "list_scope_dimension_values",
    ],
)
def test_optional_capabilities_do_not_escape_fixed_namespace(backend, operation):
    _, bound, session = backend
    args = {
        "remember": (MemoryNote(content="body"),),
        "inbox_list": (),
        "inbox_read": (["foreign-id"],),
        "delete_by_scope_dimension": ("client_application_id", "foreign"),
        "list_scope_dimension_values": ("client_application_id",),
    }[operation]
    with pytest.raises(RemoteMemoryStoreError):
        getattr(bound, operation)(*args)
    session.request.assert_not_called()


def test_native_publication_revokes_bound_remote_capability(backend):
    from xagent.web.memory_lifecycle import MemoryUnavailableError

    raw, _, session = backend
    source = Mock()
    source.publication_generation.return_value = 4
    published = RevocableMemoryStore(raw, source, 4)
    bound = bind_memory_access_scope(
        published, MemoryAccessScope(31, {"client_application_id": "42"})
    )
    session.request.return_value = reply({"results": [], "truncated": False})
    bound.search("live")
    source.publication_generation.return_value = 5
    with pytest.raises(MemoryUnavailableError):
        bound.search("revoked")
    assert session.request.call_count == 1


@pytest.mark.parametrize(
    "change", ["missing", "zero", "blank", "user", "url", "key", "mode"]
)
def test_explicit_parent_scope_failure_is_permanent(backend, monkeypatch, change):
    from xagent.core.memory.remote import delegated_agent_identity

    raw, _, session = backend
    session.request.return_value = reply({"results": [], "truncated": False})
    with delegated_agent_identity(7, store=raw):
        raw.search("parent")
        if change in {"missing", "zero", "blank", "user"}:
            current_user_id.set(
                {"missing": None, "zero": 0, "blank": " ", "user": 32}[change]
            )
        else:
            name = {"url": "SERVICE_URL", "key": "SERVICE_KEY", "mode": "AGENT_SEATS"}[
                change
            ]
            monkeypatch.setenv(
                f"XAGENT_MEMORY_{name}", "true" if change == "mode" else "changed"
            )
        with pytest.raises(RemoteMemoryStoreIdentityError):
            raw.search("changed")
        current_user_id.set(31)
        monkeypatch.setenv("XAGENT_MEMORY_SERVICE_URL", "http://memory-internal")
        monkeypatch.setenv("XAGENT_MEMORY_SERVICE_KEY", "fixture-token")
        monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "false")
        with pytest.raises(RemoteMemoryStoreIdentityError):
            raw.search("restored")
        assert session.request.call_count == 1


@pytest.mark.parametrize("change", ["url", "key"])
def test_rebinding_stale_transport_cannot_attest_current_config(
    backend, monkeypatch, change
):
    raw, _, session = backend
    monkeypatch.setenv(
        "XAGENT_MEMORY_SERVICE_URL" if change == "url" else "XAGENT_MEMORY_SERVICE_KEY",
        "http://new-internal" if change == "url" else "new-fixture-key",
    )
    fresh = bind_memory_access_scope(
        raw, MemoryAccessScope(31, {"client_application_id": "42"})
    )
    with pytest.raises(RemoteMemoryStoreIdentityError):
        fresh.list_all()
    session.request.assert_not_called()


def test_fixed_native_scope_cannot_switch_to_an_independent_agent(backend, monkeypatch):
    import xagent.core.memory.remote as remote
    from xagent.core.memory.agent_bindings import ApprovedAgentBinding

    raw, fixed, session = backend
    mapping = ApprovedAgentBinding(
        31, 7, 101, 202, 101, "http://memory-internal", "0" * 64
    )
    override = remote._ConfiguredAgentIdentityBinding(
        "31", "http://memory-internal", "0" * 64, mapping, "7"
    )
    token = remote._agent_identity_override.set(override)
    try:
        with pytest.raises(RemoteMemoryStoreIdentityError):
            fixed.search("same platform owner, different physical namespace")
    finally:
        remote._agent_identity_override.reset(token)
    with pytest.raises(RemoteMemoryStoreIdentityError):
        fixed.search("removing override cannot revive old capability")
    session.request.assert_not_called()


@pytest.mark.parametrize("change", ["url", "key", "backend"])
def test_configured_transport_rejects_drift_and_restoration(
    backend, monkeypatch, change
):
    from xagent.core.memory.remote import remote_memory_store_from_config

    _, _, session = backend
    monkeypatch.setenv("XAGENT_MEMORY_BACKEND", "remote")
    configured = remote_memory_store_from_config()
    configured._session = session
    session.request.return_value = reply({"results": [], "truncated": False})
    configured.search("before configuration drift")
    name = {"url": "SERVICE_URL", "key": "SERVICE_KEY", "backend": "BACKEND"}[change]
    monkeypatch.setenv(f"XAGENT_MEMORY_{name}", "changed")
    with pytest.raises(RemoteMemoryStoreIdentityError):
        configured.search("drift")
    monkeypatch.setenv("XAGENT_MEMORY_BACKEND", "remote")
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_URL", "http://memory-internal")
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_KEY", "fixture-token")
    with pytest.raises(RemoteMemoryStoreIdentityError):
        configured.search("restore requires publication restart")
    assert session.request.call_count == 1


@pytest.mark.parametrize("observer", ["base", "bound"])
def test_native_rebind_cannot_revive_invalidated_publication_transport(
    backend, monkeypatch, observer
):
    from xagent.core.memory.remote import remote_memory_store_from_config

    _, _, session = backend
    monkeypatch.setenv("XAGENT_MEMORY_BACKEND", "remote")
    raw = remote_memory_store_from_config()
    raw._session = session
    bound = bind_memory_access_scope(
        raw, MemoryAccessScope(31, {"client_application_id": "42"})
    )
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_KEY", "changed")
    with pytest.raises(RemoteMemoryStoreIdentityError):
        (raw if observer == "base" else bound).search("observe drift")
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_KEY", "fixture-token")
    rebound = bind_memory_access_scope(
        raw, MemoryAccessScope(31, {"client_application_id": "fresh"})
    )
    for store in (raw, bound, rebound):
        with pytest.raises(RemoteMemoryStoreIdentityError):
            store.search("publication restart is still required")
    session.request.assert_not_called()
