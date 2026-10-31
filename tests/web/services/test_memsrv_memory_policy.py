"""Remote policy identity and native namespace integration regressions."""

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from xagent.core.memory.access_scope import MemoryAccessScope
from xagent.core.memory.remote import (
    RemoteMemoryReadiness,
    RemoteMemoryStore,
    require_remote_memory_identity,
)
from xagent.core.user_context import current_user_id
from xagent.web import dynamic_memory_store, memory_lifecycle
from xagent.web.services import agent_service_manager as runtime
from xagent.web.services.memory_policy import (
    MemoryPolicyDecision,
    set_trusted_memory_policy_resolver,
)
from xagent.web.user_isolated_memory import UserContext


@pytest.fixture(autouse=True)
def remote_backend(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("XAGENT_MEMORY_BACKEND", "remote")
    set_trusted_memory_policy_resolver(None)
    yield
    set_trusted_memory_policy_resolver(None)


def _store(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock]:
    store = Mock(spec=RemoteMemoryStore)
    store.validate_identity_context.side_effect = require_remote_memory_identity
    monkeypatch.setattr(runtime, "get_memory_store", lambda: store)
    monkeypatch.setattr(dynamic_memory_store, "get_memory_store", lambda: store)
    readiness = Mock(
        return_value=RemoteMemoryReadiness("degraded", reason="embedding_unavailable")
    )
    monkeypatch.setattr(runtime, "remote_memory_readiness", readiness)
    monkeypatch.setattr(memory_lifecycle, "remote_memory_readiness", readiness)
    return store, readiness


def test_missing_ambient_identity_never_probes_remote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _store_obj, readiness = _store(monkeypatch)
    with UserContext(None):
        policy = runtime.resolve_agent_service_memory_policy(
            task=SimpleNamespace(id=17, user_id=23, agent_id=None, agent_config={})
        )
    assert not policy.memory_enabled
    assert not policy.memory_available
    readiness.assert_not_called()


@pytest.mark.asyncio
async def test_cached_policy_resolution_uses_trusted_runtime_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, readiness = _store(monkeypatch)
    manager = object.__new__(runtime.AgentServiceManager)
    agent = Mock()
    manager._agents = {17: agent}
    task = SimpleNamespace(id=17, user_id=23, agent_id=None, agent_config={})
    with UserContext(999):
        await manager._reconcile_cached_agent_memory_policy(
            17,
            task=task,
            agent_config={},
            ambient_owner_user_id=23,
            ambient_execution_scope=None,
        )
        assert current_user_id.get() == 999
    agent.revoke_memory.assert_not_called()
    call = agent.apply_memory_policy.call_args.kwargs
    assert call["memory"] is store
    assert call["enabled"] is True
    assert call["execution_metadata"]["memory_remote_status"] == "degraded"
    assert call["execution_metadata"]["memory_semantic_available"] is False
    assert readiness.called


def test_remote_cross_owner_fixed_scope_rejected_before_binding_or_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, readiness = _store(monkeypatch)
    set_trusted_memory_policy_resolver(
        lambda _request: MemoryPolicyDecision(
            enabled=True,
            available=True,
            memory_access_scope=MemoryAccessScope(700, {"principal": "other"}),
        )
    )
    with UserContext(23):
        policy = runtime.resolve_agent_service_memory_policy(
            task=SimpleNamespace(id=17, user_id=23, agent_id=None, agent_config={})
        )
    assert not policy.memory_enabled
    assert policy.memory_availability_reason == "binding_failed"
    readiness.assert_not_called()
    assert not store.validate_identity_context.called


@pytest.mark.parametrize("config,agent_id", [({}, 41), ({"is_preview": True}, None)])
def test_remote_builder_and_preview_keep_memory_disabled_without_probe(
    monkeypatch: pytest.MonkeyPatch,
    config: dict,
    agent_id: int | None,
) -> None:
    store, readiness = _store(monkeypatch)
    with UserContext(None):
        policy = runtime.resolve_agent_service_memory_policy(
            task=SimpleNamespace(
                id=17, user_id=23, agent_id=agent_id, agent_config=config
            ),
        )
    assert not policy.memory_enabled
    assert policy.memory_available
    readiness.assert_not_called()
    store.validate_identity_context.assert_not_called()


def test_unknown_backend_fences_admission_without_local_authority_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAGENT_MEMORY_BACKEND", "remtoe")
    authority = Mock(side_effect=AssertionError("local authority must not be read"))
    monkeypatch.setattr(dynamic_memory_store, "_read_authority_snapshot", authority)
    manager = dynamic_memory_store.DynamicMemoryStoreManager()
    status = manager.admit()
    assert status.state is memory_lifecycle.MemoryLifecycleState.RESTART_REQUIRED
    assert manager.acquire()[0] is None
    authority.assert_not_called()


def test_sdk_uses_authenticated_owner_memory_without_changing_builder_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, readiness = _store(monkeypatch)
    with UserContext(23):
        policy = runtime.resolve_agent_service_memory_policy(
            task=SimpleNamespace(
                id=17, user_id=23, agent_id=41, source="sdk", agent_config={}
            )
        )
    assert policy.memory is store
    assert policy.memory_enabled
    assert policy.memory_available
    assert readiness.called


@pytest.mark.parametrize("ambient", [None, 24])
def test_sdk_owner_mismatch_fails_before_any_remote_probe(monkeypatch, ambient):
    _store_obj, readiness = _store(monkeypatch)
    with UserContext(ambient):
        policy = runtime.resolve_agent_service_memory_policy(
            task=SimpleNamespace(
                id=17, user_id=23, agent_id=41, source="sdk", agent_config={}
            )
        )
    assert not policy.memory_enabled
    assert not policy.memory_available
    readiness.assert_not_called()


@pytest.mark.parametrize("preview", [False, True])
def test_sdk_respects_explicit_host_denial_and_preview_default(monkeypatch, preview):
    store, readiness = _store(monkeypatch)
    if not preview:
        set_trusted_memory_policy_resolver(
            lambda _request: MemoryPolicyDecision(enabled=False, available=True)
        )
    with UserContext(23):
        policy = runtime.resolve_agent_service_memory_policy(
            task=SimpleNamespace(
                id=17,
                user_id=23,
                agent_id=41,
                source="sdk",
                agent_config={"is_preview": preview},
            )
        )
    assert not policy.memory_enabled
    readiness.assert_not_called()
    store.validate_identity_context.assert_not_called()
