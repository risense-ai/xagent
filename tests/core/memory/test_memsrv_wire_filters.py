"""Use real requests preparation, replacing only the socket transport."""

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
import requests

from xagent.core.memory.access_scope import MemoryAccessScope, bind_memory_access_scope
from xagent.core.memory.remote import RemoteMemoryStore
from xagent.core.user_context import current_user_id


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


@pytest.mark.parametrize("operation", ["search", "list_all"])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("key", ["date_from", "date_to"])
@pytest.mark.parametrize("aware", [False, True])
def test_datetime_filters_reach_the_wire_without_losing_their_instant(
    monkeypatch: pytest.MonkeyPatch, operation: str, fixed: bool, key: str, aware: bool
) -> None:
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_URL", "http://memsrv.invalid")
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_KEY", "fixture-runtime-key")
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "false")
    session = WireSession()
    store = RemoteMemoryStore(
        "http://memsrv.invalid", "fixture-runtime-key", session=session
    )
    if fixed:
        store = bind_memory_access_scope(
            store, MemoryAccessScope(31, {"client_application_id": "42"})
        )
    boundary = datetime(
        2026,
        10,
        30,
        9,
        15,
        20,
        123456,
        tzinfo=timezone(timedelta(hours=8)) if aware else None,
    )
    filters = {key: boundary, "category": "synthetic"}
    token = current_user_id.set(31)
    try:
        if operation == "search":
            store.search("synthetic query", filters=filters)
        else:
            store.list_all(filters)
    finally:
        current_user_id.reset(token)
    assert filters == {key: boundary, "category": "synthetic"}
    assert len(session.sent) == 1
    prepared = session.sent[0]
    assert prepared.headers["X-User-External-Id"] == "31"
    wire = json.loads(prepared.body)
    encoded = datetime.fromisoformat(wire["filters"][key])
    assert encoded.utcoffset() is not None
    assert encoded.timestamp() == boundary.timestamp()
    assert wire["filters"]["category"] == "synthetic"
    if fixed:
        assert wire["filters"]["metadata"] == {
            "execution_scope_client_application_id": "42"
        }
