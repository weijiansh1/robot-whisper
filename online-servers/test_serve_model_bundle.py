from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest


MODULE_PATH = pathlib.Path(__file__).with_name("serve_model_bundle.py")
SPEC = importlib.util.spec_from_file_location("serve_model_bundle", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
bundle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bundle
SPEC.loader.exec_module(bundle)


def test_model_endpoints_are_stable_and_consecutive() -> None:
    endpoints = bundle.model_endpoints(8820)
    assert [endpoint.name for endpoint in endpoints] == [
        "goal",
        "spatial",
        "object",
        "long",
        "calvin",
    ]
    assert [endpoint.kind for endpoint in endpoints] == [
        "libero",
        "libero",
        "libero",
        "libero",
        "calvin",
    ]
    assert [endpoint.port for endpoint in endpoints] == list(range(8820, 8825))


@pytest.mark.parametrize("base_port", [0, 65532, 70000])
def test_model_endpoints_reject_invalid_port_range(base_port: int) -> None:
    with pytest.raises(ValueError, match="five consecutive ports"):
        bundle.model_endpoints(base_port)
