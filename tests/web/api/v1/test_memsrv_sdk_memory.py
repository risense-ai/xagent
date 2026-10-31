"""Authenticated SDK admission -> actual runtime memory tools -> HTTP list.

Only background scheduling, model responses and memsrv transport are synthetic.
The API-key principal, task owner, policy, ReAct tools and adapter are real.
"""

import json
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

from xagent.core.agent.service import AgentService
from xagent.core.memory.remote import RemoteMemoryReadiness, RemoteMemoryStore
from xagent.web import dynamic_memory_store, memory_lifecycle
from xagent.web.api.memory import MemoryManagementRouter
from xagent.web.models.task import Task
from xagent.web.services import agent_service_manager as runtime
from xagent.web.services.memory_policy import set_trusted_memory_policy_resolver
from xagent.web.user_isolated_memory import UserContext

from ..conftest import (
    _admin_headers,
    _direct_db_session,
    _register_second_user,
    app_for_tests,
    client,
)

pytestmark = pytest.mark.usefixtures("_test_db")
FACT = "Synthetic SDK preference: use indigo quarterly charts."


class MemorySession:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []
        self.notes: dict[str, dict[str, Any]] = {}

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        identity = kwargs["headers"]["X-User-External-Id"]
        body = kwargs.get("json") or {}
        if method == "POST" and url.endswith("/notes"):
            note = {**body, "id": "sdk-note", "owner_external_id": identity}
            self.notes[note["id"]] = note
            payload = {"id": note["id"], "success": True}
        else:
            assert method == "POST" and url.endswith(("/list", "/search"))
            category = (body.get("filters") or {}).get("category")
            notes = [
                note
                for note in self.notes.values()
                if note["owner_external_id"] == identity
                and (category is None or note["category"] == category)
            ]
            payload = {"results": notes, "truncated": False}
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(payload).encode()
        return response


class ScriptedLLM:
    model_name = "synthetic-no-network"

    def __init__(self, tool, arguments):
        self.tool, self.arguments = tool, arguments
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) > 1:
            return {"content": "Synthetic response", "done": True}
        return {
            "content": "",
            "done": False,
            "tool_calls": [
                {
                    "id": "synthetic-call",
                    "function": {
                        "name": self.tool,
                        "arguments": json.dumps(self.arguments),
                    },
                }
            ],
        }


@pytest.mark.asyncio
async def test_sdk_fact_persists_is_listed_and_is_recalled_by_next_task(monkeypatch):
    monkeypatch.setenv("XAGENT_MEMORY_BACKEND", "remote")
    monkeypatch.setattr(
        "xagent.web.services.task_orchestrator._schedule_bg", MagicMock()
    )
    set_trusted_memory_policy_resolver(None)
    session = MemorySession()
    store = RemoteMemoryStore(
        "http://memsrv.invalid",
        "synthetic-runtime-key",
        session=cast(requests.Session, session),
    )
    monkeypatch.setattr(runtime, "get_memory_store", lambda: store)
    monkeypatch.setattr(dynamic_memory_store, "get_memory_store", lambda: store)
    for module in (runtime, memory_lifecycle):
        monkeypatch.setattr(
            module,
            "remote_memory_readiness",
            lambda: RemoteMemoryReadiness("ready", semantic_available=True),
        )

    headers = _admin_headers()
    created = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": "synthetic SDK",
            "instructions": "Use memory tools for durable preferences.",
            "execution_mode": "balanced",
        },
    )
    assert created.status_code == 200, created.text
    agent_id = created.json()["id"]
    key = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)
    assert key.status_code == 200, key.text
    sdk_headers = {"Authorization": "Bearer " + key.json()["full_key"]}

    def create_task(text):
        admitted = client.post(
            "/v1/chat/tasks",
            headers=sdk_headers,
            json={
                "agent_id": agent_id,
                "message": {"role": "user", "content": text},
            },
        )
        assert admitted.status_code == 202, admitted.text
        with _direct_db_session() as db:
            task = db.query(Task).filter(Task.id == admitted.json()["task_id"]).one()
            db.expunge(task)
        assert task.source == "sdk" and task.agent_id == agent_id
        return task

    async def execute(task, llm):
        with UserContext(task.user_id):
            policy = await runtime.resolve_agent_service_memory_policy_async(task=task)
            assert policy.memory_enabled and policy.memory_available
            service = AgentService(
                name="synthetic SDK",
                id=str(task.id),
                pattern="react",
                tools=[],
                llm=cast(Any, llm),
                memory=policy.memory,
                memory_enabled=policy.memory_enabled,
                memory_similarity_threshold=0.4,
                enable_workspace=False,
                enable_default_tools=False,
                skills_enabled=False,
                execution_metadata=policy.execution_metadata(
                    ambient_owner_user_id=task.user_id,
                    ambient_execution_scope=None,
                ),
            )
            result = await service.execute_task(task.input, task_id=str(task.id))
            assert result["success"] is True

    first = create_task("Remember " + FACT)
    writer = ScriptedLLM("store_memory", {"content": FACT, "kind": "user_preference"})
    await execute(first, writer)
    assert session.notes["sdk-note"]["content"] == FACT
    transcript = str(writer.calls)
    assert "store_memory" in transcript and "Memory persistence:" in transcript
    assert "stored" in transcript

    memory_app = FastAPI()
    memory_app.include_router(MemoryManagementRouter(lambda: store).get_router())
    memory_app.dependency_overrides.update(app_for_tests.dependency_overrides)
    with TestClient(memory_app) as memory_client:
        listed = memory_client.get("/api/memory/list", headers=headers)
        other = memory_client.get("/api/memory/list", headers=_register_second_user())
    assert listed.status_code == 200, listed.text
    assert [note["content"] for note in listed.json()["memories"]] == [FACT]
    assert other.status_code == 200 and other.json()["memories"] == []

    second = create_task("Recall my chart preference")
    reader = ScriptedLLM("search_memory", {"query": "chart preference"})
    await execute(second, reader)
    assert FACT in str(reader.calls)
    assert any(call["url"].endswith("/search") for call in session.calls)
    # Top-level SDK tasks use their authenticated user's memory. Independent
    # AgentTool delegations retain the separate verified seat + owner fence.
    # Only the authenticated second user's list is routed to another namespace.
    own_calls = [call for call in session.calls if not call["url"].endswith("/list")]
    assert all(
        call["headers"]["X-User-External-Id"] == str(first.user_id)
        for call in own_calls
    )
    assert all("X-Memory-Owner-Id" not in call["headers"] for call in session.calls)
