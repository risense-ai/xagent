"""Actual memory tool + remote adapter with synthetic scored service replies."""

import json
from typing import Any, cast

import pytest
import requests

from xagent.core.agent.context.memory_tool import SearchMemoryTool, StoreMemoryTool
from xagent.core.agent.service import AgentService
from xagent.core.memory.remote import RemoteMemoryStore
from xagent.web.user_isolated_memory import UserContext


class ScoredSession:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        assert kwargs["headers"]["X-User-External-Id"] == "31"
        if url.endswith("/search"):
            # Mirrors memsrv's lexical fallback: a lexical hit is not limited
            # by similarity_threshold (that field controls vector distance).
            payload = {"results": self.hits}
        else:
            assert method == "POST" and url.endswith("/notes")
            payload = {"id": kwargs["json"]["id"], "success": True}
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(payload).encode()
        return response


async def store_with_hits(hits, content="Synthetic preference: use indigo charts."):
    session = ScoredSession(hits)
    store = RemoteMemoryStore(
        "http://memsrv.invalid",
        "synthetic-runtime-key",
        session=cast(requests.Session, session),
    )
    with UserContext(31):
        result = await StoreMemoryTool(memory_store=store, task="synthetic").execute(
            content
        )
    return result, session


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "score", [0.12, 0.65, 0.899, None, "0.99", True, float("nan"), float("inf"), 1.1]
)
async def test_low_or_unverifiable_hit_does_not_discard_a_new_fact(score):
    result, session = await store_with_hits(
        [
            {
                "id": "other",
                "content": "Synthetic preference: blue weekly tables.",
                "category": "react_memory",
                "score": score,
                "fused": 0.99,
            }
        ]
    )
    assert result["stored"] is True
    assert [call["url"].rsplit("/", 1)[-1] for call in session.calls] == [
        "search",
        "notes",
    ]
    assert session.calls[0]["json"]["filters"] == {"category": "react_memory"}
    assert session.calls[0]["json"]["similarity_threshold"] == 0.9
    assert "similarity_score" not in session.calls[-1]["json"]


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["", "  ", None])
async def test_empty_high_scoring_hit_cannot_prove_a_duplicate(content):
    result, _session = await store_with_hits(
        [{"id": "empty", "content": content, "score": 1.0}]
    )
    assert result["stored"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("score", [None, 0.1, 0.99])
async def test_exact_duplicate_still_skips_the_write(score):
    content = "Synthetic exact repeat."
    result, session = await store_with_hits(
        [{"id": "same", "content": content, "score": score}], content
    )
    assert result["success"] is True and result["stored"] is False
    assert len(session.calls) == 1


@pytest.mark.asyncio
async def test_high_scoring_different_fact_is_not_proof_of_duplication():
    result, session = await store_with_hits(
        [
            {
                "id": "near",
                "content": "Synthetic preference: use indigo tables, not charts.",
                "score": 0.99,
            }
        ]
    )
    assert result["stored"] is True
    assert len(session.calls) == 2


@pytest.mark.asyncio
async def test_empty_results_store_and_missing_identity_never_calls_http():
    result, session = await store_with_hits([])
    assert result["stored"] is True and len(session.calls) == 2
    session = ScoredSession([])
    store = RemoteMemoryStore(
        "http://memsrv.invalid",
        "synthetic-runtime-key",
        session=cast(requests.Session, session),
    )
    with UserContext(None):
        result = await StoreMemoryTool(memory_store=store, task="synthetic").execute(
            "Synthetic unique fact."
        )
    assert result["success"] is False
    assert session.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime_chain", [False, True])
async def test_two_similar_new_facts_persist_recall_and_exact_repeat_deduplicates(
    runtime_chain,
):
    class StatefulSession(ScoredSession):
        def __init__(self):
            super().__init__([])
            self.notes = []

        def request(self, method, url, **kwargs):
            body = kwargs.get("json") or {}
            if url.endswith("/search"):
                self.hits = sorted(
                    self.notes,
                    key=lambda note: note["content"] == body["query"],
                    reverse=True,
                )[: body["k"]]
            response = super().request(method, url, **kwargs)
            if url.endswith("/notes"):
                self.notes.append({**body, "score": 0.99})
            return response

    session = StatefulSession()
    adapter = RemoteMemoryStore(
        "http://memsrv.invalid",
        "synthetic-runtime-key",
        session=cast(requests.Session, session),
    )
    tool = StoreMemoryTool(memory_store=adapter, task="synthetic")
    facts = [
        "SYNTHETIC_A: chart shade is indigo.",
        "SYNTHETIC_B: chart shade is amber.",
    ]
    with UserContext(31):
        if runtime_chain:
            commands = [
                ("store_memory", {"content": facts[0]}),
                ("store_memory", {"content": facts[1]}),
                ("store_memory", {"content": facts[0]}),
                ("search_memory", {"query": "chart shade"}),
            ]

            class ScriptedLLM:
                model_name = "synthetic-no-network"

                def __init__(self):
                    self.calls = []

                async def chat(self, **kwargs):
                    self.calls.append(kwargs)
                    index = len(self.calls) - 1
                    if index == len(commands):
                        return {"content": "Synthetic completion", "done": True}
                    name, args = commands[index]
                    return {
                        "content": "",
                        "done": False,
                        "tool_calls": [
                            {
                                "id": f"synthetic-{index}",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(args),
                                },
                            }
                        ],
                    }

            llm = ScriptedLLM()
            service = AgentService(
                name="synthetic dedup",
                id="synthetic-dedup",
                pattern="react",
                llm=cast(Any, llm),
                memory=adapter,
                memory_enabled=True,
                tools=[],
                enable_workspace=False,
                enable_default_tools=False,
                skills_enabled=False,
            )
            result = await service.execute_task(
                "Remember and recall the synthetic chart facts."
            )
            assert result["success"] is True
            assert all(fact in str(llm.calls[-1]) for fact in facts)
        else:
            for fact in facts:
                assert (await tool.execute(fact))["stored"] is True
            assert (await tool.execute(facts[0]))["stored"] is False
        recalled = await SearchMemoryTool(memory_store=adapter).execute("chart shade")
    assert {note["content"] for note in recalled["memories"]} == set(facts)
    assert len(session.notes) == 2
    assert sum(call["url"].endswith("/notes") for call in session.calls) == 2
