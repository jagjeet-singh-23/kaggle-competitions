"""Golden tier: real entry points run in subprocesses and pinned byte for byte.

Run with `pytest -m golden`. Set GOLDEN_UPDATE=1 to record missing manifest
digests and per-case wall times. Anchored cases always assert their constants.
"""

import json
import os
import re
import signal
import time
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


def test_timeout_kills_whole_process_tree(tmp_path):
    log = tmp_path / "tree.log"
    cmd = ["sh", "-c", "sleep 60 & echo child=$!; wait"]
    with pytest.raises(RuntimeError, match="timed out after 1s") as exc:
        golden.run_in_new_session(cmd, log, "tree", 1)
    child = int(re.search(r"child=(\d+)", str(exc.value)).group(1))
    for _ in range(50):  # the kernel reaps the killed child asynchronously
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    os.kill(child, signal.SIGKILL)
    pytest.fail("child of the timed-out process survived")


@pytest.mark.golden
@pytest.mark.parametrize("name", list(golden.CASES))
def test_golden_case(name):
    update = os.environ.get("GOLDEN_UPDATE") == "1"
    digest, minutes = golden.timed_run_case(name)
    manifest = _load()
    where = f"{name}: output {golden.output_path(name)}, log {golden.OUT_DIR / (name + '.log')}"
    if update:
        # Minutes are recorded only with a newly recorded digest, so an update run
        # that records nothing leaves the manifest byte-identical.
        if name not in ANCHORS and name not in manifest:
            manifest[name] = digest
            manifest.setdefault("_minutes", {})[name] = minutes
            _save(manifest)
    if name in ANCHORS:
        assert digest == ANCHORS[name], f"digest differs from anchor for {where}"
    else:
        assert name in manifest, f"{name} has no manifest entry; run with GOLDEN_UPDATE=1"
        assert digest == manifest[name], f"digest differs from manifest for {where}"
