"""Golden tier: real entry points run in subprocesses and pinned byte for byte.

Run with `pytest -m golden`. Set GOLDEN_UPDATE=1 to record missing manifest
digests and per-case wall times. Anchored cases always assert their constants.
"""

import json
import os
from pathlib import Path

import golden
import pytest

MANIFEST = Path(__file__).with_name("golden_manifest.json")

# Digests of files that shipped, hard-coded so GOLDEN_UPDATE cannot overwrite them.
ANCHORS = {
    # raasta-b-c1 sub_w16.csv, v17's tested file
    "submit_v17": "2fa67eea992ea4147ea68f85def0d2c4aa9ac7f177697f6a90174f91a2e4c5c9",
    # raasta-b-field sub_v19_on.csv
    "submit_v19": "2a6cc40956b120c2969ae6899b97977c9ea2e5fca867b9485f9009d8551894e4",
}


def _load() -> dict:
    return json.loads(MANIFEST.read_text())


def _save(manifest: dict) -> None:
    MANIFEST.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")


def test_golden_env_is_minimal():
    env = golden.golden_env(X="1")
    assert set(env) == {"PATH", "HOME", "MIN_AVAIL_MB", "X"}


def test_golden_cases_shape():
    assert set(golden.CASES) == {"submit_v17", "submit_v19", "submit_rankby", "dilute_v17"}
    env = golden.case_env("dilute_v17")
    assert not set(golden.SUBMIT_ONLY_KEYS) & set(env)
    assert env["CASMI_DATA"] == golden.D
    assert env["PYTHONPATH"] == str(golden.MOD)


@pytest.mark.golden
@pytest.mark.parametrize("name", list(golden.CASES))
def test_golden_case(name):
    update = os.environ.get("GOLDEN_UPDATE") == "1"
    digest, minutes = golden.timed_run_case(name)
    manifest = _load()
    if update:
        manifest.setdefault("_minutes", {})[name] = minutes
        if name not in ANCHORS and name not in manifest:
            manifest[name] = digest
        _save(manifest)
    if name in ANCHORS:
        assert digest == ANCHORS[name]
    else:
        assert name in manifest, f"{name} has no manifest entry; run with GOLDEN_UPDATE=1"
        assert digest == manifest[name]
