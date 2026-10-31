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
            if not url.endswith("/whoami") and state.get("data_handler") is not None:
                return state["data_handler"](method, url, **kwargs)
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

    real_service = service.AgentService
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
            engine=engine,
            factory=factory,
            calls=calls,
            store=store,
            state=state,
            real_service=real_service,
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
async def test_real_delegated_memory_tools_confine_inherited_acl_to_agent_owner(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run actual AgentService/ReAct/tools/to_thread against synthetic HTTP.

    memsrv agents inherit their person's ACL. Acting as agent:7 alone may
    read/write the parent and sibling notes; its owner header narrows that
    existing ACL server-side. No real LLM, service, or user content is used.
    """
    from xagent.core.agent import service
    from xagent.core.tools.adapters.vibe import agent_model_resolution
    from xagent.web.user_isolated_memory import UserContext

    notes = {
        "parent-note": {
            "id": "parent-note",
            "owner_id": 101,
            "content": "SYNTHETIC_PARENT_PRIVATE",
            "category": "react_memory",
        },
        "sibling-note": {
            "id": "sibling-note",
            "owner_id": 303,
            "content": "SYNTHETIC_SIBLING_PRIVATE",
            "category": "react_memory",
        },
        "company-note": {
            "id": "company-note",
            "owner_id": 404,
            "content": "SYNTHETIC_COMPANY_SHARED",
            "category": "react_memory",
            "space_kind": "company",
        },
    }
    original_notes = json.loads(json.dumps(notes))

    def respond(status: int, payload: Any) -> requests.Response:
        response = requests.Response()
        response.status_code = status
        response._content = json.dumps(payload).encode()
        return response

    def memory_http(method: str, url: str, **kwargs: Any) -> requests.Response:
        headers = kwargs["headers"]
        selected = headers.get("X-Memory-Owner-Id")
        # Mirrors memsrv owner(): explicit selection intersects the derived
        # inherited ACL and narrows both reads and writes, clearing spaces.
        owners = {101, 202, 303, 404} if selected is None else {int(selected)}
        body = kwargs.get("json") or {}
        if url.endswith("/search"):
            category = (body.get("filters") or {}).get("category")
            hits = [
                dict(note)
                for note in notes.values()
                if note["owner_id"] in owners
                and (category is None or note["category"] == category)
            ]
            return respond(200, {"results": hits[: body.get("k", 5)]})
        if method == "POST" and url.endswith("/notes"):
            note = {**body, "id": "agent-note", "owner_id": int(selected or 202)}
            notes["agent-note"] = note
            return respond(200, {"id": "agent-note", "success": True})
        note_id = url.rsplit("/", 1)[-1]
        note = notes.get(note_id)
        if note is None or note["owner_id"] not in owners:
            return respond(404, {"error": "not found"})
        if method == "GET":
            return respond(200, note)
        if method == "PATCH":
            note.update(body)
            return respond(
                200, {"id": note_id, "success": True, "owner_id": note["owner_id"]}
            )
        assert method == "DELETE"
        del notes[note_id]
        return respond(200, {"id": note_id, "success": True})

    runtime.state["data_handler"] = memory_http
    commands = [
        ("store_memory", {"content": "Synthetic agent prefers teal reports."}),
        ("search_memory", {"query": "synthetic preference"}),
        ("update_memory", {"memory_id": "parent-note", "content": "forbidden"}),
        ("delete_memory", {"memory_id": "parent-note"}),
        ("update_memory", {"memory_id": "sibling-note", "content": "forbidden"}),
        ("delete_memory", {"memory_id": "sibling-note"}),
        (
            "update_memory",
            {
                "memory_id": "agent-note",
                "content": "Synthetic updated teal preference.",
            },
        ),
        ("search_memory", {"query": "synthetic updated preference"}),
        ("delete_memory", {"memory_id": "agent-note"}),
    ]

    class ScriptedLLM:
        model_name = "synthetic-no-network"

        def __init__(self) -> None:
            self.calls: list[str] = []

        async def chat(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(str(kwargs))
            index = len(self.calls) - 1
            if index >= len(commands):
                return {"content": "synthetic answer", "done": True}
            name, arguments = commands[index]
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": f"synthetic-{index}",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                ],
                "done": False,
            }

    llm = ScriptedLLM()

    class ActualService(runtime.real_service):
        def __init__(self, **kwargs: Any) -> None:
            kwargs.update(
                tools=[],
                tool_config=None,
                enable_workspace=False,
                enable_default_tools=False,
                skills_enabled=False,
            )
            super().__init__(**kwargs)

    monkeypatch.setattr(service, "AgentService", ActualService)
    monkeypatch.setattr(
        agent_model_resolution,
        "resolve_agent_model_llms",
        lambda *_args: (llm, None, None, None),
    )
    result = await _run(runtime, monkeypatch)
    assert result == {"response": "synthetic answer"}
    assert len(llm.calls) == len(commands) + 1
    transcript = "\n".join(llm.calls)
    assert "SYNTHETIC_PARENT_PRIVATE" not in transcript
    assert "SYNTHETIC_SIBLING_PRIVATE" not in transcript
    assert "SYNTHETIC_COMPANY_SHARED" not in transcript
    assert "Synthetic updated teal preference." in transcript
    assert "'stored': True" in transcript or '"stored": true' in transcript
    assert notes == original_notes  # only the created agent note was deleted
    data_calls = [call for call in runtime.calls if not call["url"].endswith("/whoami")]
    assert data_calls
    assert all(
        call["headers"]["X-User-External-Id"] == "agent:7" for call in data_calls
    )
    assert all(call["headers"]["X-Memory-Owner-Id"] == "202" for call in data_calls)
    assert any(call["method"] == "PATCH" for call in data_calls)
    assert any(call["method"] == "DELETE" for call in data_calls)
    assert (
        sum(
            call["method"] == "POST" and call["url"].endswith("/notes")
            for call in data_calls
        )
        == 1
    )
    assert sum(call["method"] == "PATCH" for call in data_calls) == 1
    assert sum(call["method"] == "DELETE" for call in data_calls) == 1
    with UserContext(31):
        runtime.store.search("ordinary parent after delegation")
    assert runtime.calls[-1]["headers"]["X-User-External-Id"] == "31"
    assert "X-Memory-Owner-Id" not in runtime.calls[-1]["headers"]


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
@pytest.mark.parametrize("parent_source", [None, "internal", "sdk", "trigger"])
def test_delegated_memory_honors_trusted_host_policy(
    runtime, policy_kind, parent_source
):
    from contextlib import nullcontext

    from xagent.core.memory.access_scope import MemoryAccessScope
    from xagent.core.tools.adapters.vibe.mcp_approval_gate import (
        ToolCallExecutionContext,
        bind_tool_call_execution_context,
    )
    from xagent.web.services.memory_policy import (
        MemoryPolicyDecision,
        set_trusted_memory_policy_resolver,
    )
    from xagent.web.user_isolated_memory import UserContext

    def policy(request):
        assert request.source == parent_source
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
        parent_context = (
            nullcontext()
            if parent_source is None
            else bind_tool_call_execution_context(
                ToolCallExecutionContext(
                    task_source=parent_source,
                    task_id="17",
                    run_id="synthetic-run",
                    turn_id="synthetic-turn",
                    tool_call_id="synthetic-call",
                    pattern="react",
                    react_step_id="synthetic-step",
                )
            )
        )
        with UserContext(31), parent_context:
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
