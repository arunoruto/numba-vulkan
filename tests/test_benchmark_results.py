"""Collected benchmark results (benchmarks/results/) follow the file format.

The format is described in benchmarks/collect.py. These checks need no GPU;
they keep contributed files readable by benchmarks/report.py.
"""

import json
import math
import pathlib
import re

import pytest

RESULTS = pathlib.Path(__file__).parent.parent / "benchmarks" / "results"
FILES = sorted(RESULTS.glob("*/*.json"))
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
TOP = {
    "schema": int,
    "user": str,
    "machine": str,
    "date": str,
    "git": dict,
    "software": dict,
    "hardware": dict,
    "settings": dict,
    "results": list,
}
RECORD = {
    "suite": str,
    "workload": str,
    "description": str,
    "backend": str,
    "variant": str,
    "label": str,
    "samples_s": list,
    "check": str,
}


@pytest.mark.skipif(not FILES, reason="no collected results")
@pytest.mark.parametrize(
    "path", FILES, ids=[f"{p.parent.name}/{p.name}" for p in FILES]
)
def test_results_file_follows_the_format(path):
    data = json.loads(path.read_text())
    for key, kind in TOP.items():
        assert isinstance(data.get(key), kind), key
    assert data["schema"] == 1
    assert NAME.match(data["user"]) and NAME.match(data["machine"])
    assert DATE.match(data["date"])
    # The path is derived from the content: results/<user>/<date>.json
    assert path.parent.name == data["user"]
    assert path.name == data["date"].replace(":", "") + ".json"
    assert set(data["git"]) >= {"commit", "branch", "dirty"}
    assert isinstance(data["hardware"].get("cpu"), dict)
    assert isinstance(data["hardware"].get("vulkan"), list)
    assert data["results"], "no results"
    for record in data["results"]:
        for key, kind in RECORD.items():
            assert isinstance(record.get(key), kind), key
        assert record["suite"] in ("apps", "kernels")
        assert record["backend"] in ("cpu", "vulkan", "cuda")
        assert (record["device"] is None) == (record["backend"] == "cpu")
        assert record["first_s"] is None or record["first_s"] > 0
        assert record["samples_s"]
        assert all(
            isinstance(t, float) and math.isfinite(t) and t > 0
            for t in record["samples_s"]
        )


def test_no_stray_files_in_results():
    if not RESULTS.exists():
        pytest.skip("no collected results")
    stray = [
        p
        for p in RESULTS.rglob("*")
        if p.is_file() and not (p.suffix == ".json" and p.parent.parent == RESULTS)
    ]
    assert not stray, f"only results/<user>/<date>.json belongs here: {stray}"
