from __future__ import annotations

import pytest

from flow_query_ids import query_id_at


def test_query_base_separates_client_invocations() -> None:
    first = [query_id_at(0, step) for step in range(7)]
    second = [query_id_at(7, step) for step in range(4)]
    assert first == list(range(7))
    assert second == list(range(7, 11))
    assert set(first).isdisjoint(second)


def test_query_id_rejects_negative_axes() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        query_id_at(-1, 0)
    with pytest.raises(ValueError, match="non-negative"):
        query_id_at(0, -1)
