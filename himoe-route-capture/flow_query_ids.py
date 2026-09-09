"""Pure query-ID helpers shared by flow-trace rollout clients."""

from __future__ import annotations


def query_id_at(query_base: int, control_step: int) -> int:
    """Return the invocation-global query ID for one local control step."""
    if query_base < 0 or control_step < 0:
        raise ValueError("query_base and control_step must be non-negative")
    return query_base + control_step
