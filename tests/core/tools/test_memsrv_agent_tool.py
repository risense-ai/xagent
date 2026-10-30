"""Delegated memory uses personal ownership, independently of execution ACLs.

The SQL ownership reads are real; memsrv HTTP and the child LLM runtime are
synthetic. No model calls or network requests leave this test process.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any

import pytest
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import xagent.core.tools.adapters.vibe.agent_tool as module
from xagent.core.memory import remote
from xagent.core.memory.remote import RemoteMemoryStore
from xagent.core.tools.adapters.vibe.agent_tool import AgentTool
from xagent.core.user_context import current_user_id
from xagent.web.models import Base, User
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.services import agent_team_scope

URL = "http://memory-internal:8080"
KEY = "synthetic-runtime-key"


@pytest.fixture
def runtime(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    engine = create_engine(f"sqlite:///{tmp_path / 'delegation.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        db.add_all(
            [
                User(id=31, username="owner", password_hash="fixture"),
                User(id=32, username="other", password_hash="fixture"),
            ]
        )
        db.add(
            Agent(
                id=7,
                user_id=31,
                name="Synthetic delegate",
                instructions="Keep official instructions",
                status=AgentStatus.PUBLISHED,
                execution_mode="balanced",
                models={"general": 1},
                knowledge_bases=[],
                skills=[],
                tool_categories=[],
                team_id=10,
                visibility="team",
            )
        )
        db.commit()
    monkeypatch.setenv("XAGENT_MEMORY_BACKEND", "remote")
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_URL", URL)
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_KEY", KEY)
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "true")
    binding = {
        "user_id": 31,
        "agent_id": 7,
        "memory_user_id": 101,
        "memory_agent_id": 202,
        "memory_agent_owner_id": 101,
        "service_url": URL,
        "runtime_profile": hashlib.sha256(KEY.encode()).hexdigest(),
    }
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_BINDINGS", json.dumps([binding]))
    calls: list[dict[str, Any]] = []

    class Session(requests.Session):
        def request(self, *args: Any, **kwargs: Any) -> requests.Response:
            method, url = args
            calls.append({"method": method, "url": url, **kwargs})
            actor = kwargs["headers"]["X-User-External-Id"]
            assert actor in {"31", "agent:7"}
            payload = (
                {
                    "scopes": ["memory:read", "memory:write"],
                    "identity": {"id": 101 if actor == "31" else 202},
                }
                if url.endswith("/whoami")
                else {"results": []}
            )
            response = requests.Response()
            response.status_code = 200
            response._content = json.dumps(payload).encode()
            return response

    store = RemoteMemoryStore(URL, KEY, session=Session())
    state: dict[str, Any] = {"constructed": [], "probes": [], "mutate": None}

    def ready(**_kwargs: Any) -> tuple[bool, None]:
        state["probes"].append(current_user_id.get())
        return True, None

    monkeypatch.setattr(remote, "remote_memory_ready", ready)
    from xagent.web import dynamic_memory_store

    monkeypatch.setattr(dynamic_memory_store, "get_memory_store", lambda: store)

    class RecordingService:
        workspace = None

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            state["constructed"].append(kwargs)

        async def execute_task(self, **kwargs: Any) -> dict[str, str]:
            assert current_user_id.get() == 31
            assert kwargs["context"]["system_prompt"].startswith(
                "Keep official instructions"
            )
            if state["mutate"] is not None:
                state["mutate"]()
            if self.kwargs["memory_enabled"]:
                self.kwargs["memory"].search("synthetic delegated query")
            return {"output": "synthetic answer"}

    from xagent.core.agent import service
    from xagent.core.tools.adapters.vibe import agent_model_resolution

    monkeypatch.setattr(service, "AgentService", RecordingService)
    monkeypatch.setattr(
        agent_model_resolution,
        "resolve_agent_model_llms",
        lambda *_args: (object(), None, None, None),
    )
    monkeypatch.setattr(
        module, "WebToolConfig", lambda **_kwargs: SimpleNamespace(close=lambda: None)
    )
    token = current_user_id.set(999)
    try:
        yield SimpleNamespace(
            engine=engine, factory=factory, calls=calls, store=store, state=state
        )
    finally:
        current_user_id.reset(token)
        engine.dispose()


async def _run(runtime: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    tool = AgentTool(
        agent_id=7,
        agent_name="Synthetic delegate",
        agent_description="synthetic",
        session_factory=runtime.factory,
        user_id=31,
        tool_name="synthetic_delegate",
        tool_description="synthetic",
    )
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)
    result = await tool.run_json_async({"task": "synthetic task"})
    assert current_user_id.get() == 999
    return result


@pytest.mark.asyncio
async def test_owned_binding_uses_independent_identity(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = await _run(runtime, monkeypatch)
    assert result == {"response": "synthetic answer"}
    child = runtime.state["constructed"][0]
    assert child["memory"] is runtime.store
    assert child["memory_enabled"] and child["memory_available"]
    assert runtime.state["probes"] == [31]
    assert [call["headers"]["X-User-External-Id"] for call in runtime.calls] == [
        "31",
        "agent:7",
        "agent:7",
    ]
    assert not any("/org/" in call["url"] for call in runtime.calls)


@pytest.mark.asyncio
async def test_team_executable_foreign_agent_gets_no_memory(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    with runtime.factory() as db:
        db.query(Agent).filter(Agent.id == 7).update({Agent.user_id: 32})
        db.commit()
    monkeypatch.setattr(
        agent_team_scope,
        "get_agent_team_scope",
        lambda *_args: agent_team_scope.AgentTeamScope(team_id=10, is_team_admin=False),
    )
    result = await _run(runtime, monkeypatch)
    assert result == {"response": "synthetic answer"}
    child = runtime.state["constructed"][0]
    assert child["memory"] is None
    assert not child["memory_enabled"] and not child["memory_available"]
    assert child["memory_availability_reason"] == "agent_memory_ownership_required"
    assert runtime.state["probes"] == []
    assert runtime.calls == []


@pytest.mark.asyncio
async def test_missing_mapping_never_probes_or_falls_back(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_BINDINGS", "[]")
    assert await _run(runtime, monkeypatch) == {"response": "synthetic answer"}
    assert runtime.state["constructed"][0]["memory"] is None
    assert runtime.state["probes"] == []
    assert runtime.calls == []


@pytest.mark.asyncio
async def test_local_backend_keeps_one_shot_memory_disabled(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAGENT_MEMORY_BACKEND", "lancedb")
    assert await _run(runtime, monkeypatch) == {"response": "synthetic answer"}
    child = runtime.state["constructed"][0]
    assert not child["memory_enabled"] and child["memory"] is None
    assert child["memory_available"] is True
    assert child["memory_availability_reason"] is None
    assert runtime.state["probes"] == []
    assert runtime.calls == []


@pytest.mark.asyncio
async def test_explicit_parent_mode_retains_person_identity(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "false")
    assert await _run(runtime, monkeypatch) == {"response": "synthetic answer"}
    assert runtime.state["constructed"][0]["memory_enabled"]
    assert [call["headers"]["X-User-External-Id"] for call in runtime.calls] == ["31"]


@pytest.mark.asyncio
async def test_ownership_change_during_child_execution_denies_memory(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def revoke() -> None:
        with runtime.factory() as db:
            db.query(Agent).filter(Agent.id == 7).update({Agent.user_id: 32})
            db.commit()

    runtime.state["mutate"] = revoke
    result = await _run(runtime, monkeypatch)
    assert result["success"] is False
    assert "agent_memory_ownership_required" in result["response"]
    assert all(call["url"].endswith("/whoami") for call in runtime.calls)


@pytest.mark.asyncio
async def test_failed_readiness_disables_memory_with_reason(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        remote, "remote_memory_ready", lambda: (False, "remote_dependency_unavailable")
    )
    assert await _run(runtime, monkeypatch) == {"response": "synthetic answer"}
    child = runtime.state["constructed"][0]
    assert child["memory"] is None
    assert child["memory_availability_reason"] == "remote_dependency_unavailable"
    assert runtime.calls == []


@pytest.mark.parametrize("policy_kind", ["disabled", "fixed", "failed"])
def test_delegated_memory_honors_trusted_host_policy(runtime, policy_kind):
    from xagent.core.memory.access_scope import MemoryAccessScope
    from xagent.web.services.memory_policy import (
        MemoryPolicyDecision,
        set_trusted_memory_policy_resolver,
    )
    from xagent.web.user_isolated_memory import UserContext

    def policy(request):
        assert request.source == "agent_tool"
        if request.task_id != 17:
            return MemoryPolicyDecision(True, True)
        assert request.user_id == 31 and request.agent_id == 7
        if policy_kind == "failed":
            raise RuntimeError("synthetic host failure")
        if policy_kind == "disabled":
            return MemoryPolicyDecision(False, True, "host_disabled")
        return MemoryPolicyDecision(
            True,
            True,
            memory_access_scope=MemoryAccessScope(31, {"client_application_id": "42"}),
        )

    set_trusted_memory_policy_resolver(policy)
    try:
        with UserContext(31):
            store, _available, reason = module._resolve_delegated_parent_memory(
                user_id=31, agent_id=7, task_id=17
            )
        assert store is None
        assert (
            reason
            == {
                "disabled": "host_disabled",
                "failed": "resolver_failed",
                "fixed": "delegated_memory_fixed_scope_unsupported",
            }[policy_kind]
        )
        assert runtime.state["probes"] == []
        assert runtime.calls == []
    finally:
        set_trusted_memory_policy_resolver(None)
