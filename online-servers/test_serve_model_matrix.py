from __future__ import annotations

import argparse
import pathlib
import sys

import pytest


MODULE_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(MODULE_DIR))
import serve_model_matrix as matrix  # noqa: E402


def test_parse_gpus() -> None:
    assert matrix.parse_gpus("0,2,7") == (0, 2, 7)


@pytest.mark.parametrize("value", ["", "0,0", "8", "gpu0"])
def test_parse_gpus_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        matrix.parse_gpus(value)


def test_matrix_endpoints_preserve_physical_ports_and_logical_order() -> None:
    records = matrix.matrix_endpoints((2, 7), 8800)
    assert len(records) == 10
    assert (records[0].physical_gpu, records[0].logical_gpu) == (2, 0)
    assert [record.model.port for record in records[:5]] == list(range(8820, 8825))
    assert (records[5].physical_gpu, records[5].logical_gpu) == (7, 1)
    assert [record.model.port for record in records[5:]] == list(range(8870, 8875))
