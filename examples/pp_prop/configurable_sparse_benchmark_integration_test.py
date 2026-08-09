"""End-to-end test for the configurable sparse pp-prop benchmark."""

import json
import pathlib
import subprocess
import sys


SCRIPT = pathlib.Path(__file__).with_name("16-configurable-sparse-benchmark.py")


def test_tiny_supervised_worker_emits_learning_schema():
    command = [
        sys.executable,
        str(SCRIPT),
        "--neurons",
        "12",
        "--degree",
        "3",
        "--steps",
        "3",
        "--final-window",
        "1",
        "--updates",
        "1",
        "--max-rss-gib",
        "4",
        "--min-available-gib",
        "1",
        "--max-wall-seconds",
        "120",
    ]

    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=150,
    )
    payload = json.loads(completed.stdout)

    assert payload["schema_version"] == 1
    assert payload["status"] == "completed"
    assert payload["metrics"]["recurrent_nnz"] == 36
    assert payload["metrics"]["updates_completed"] == 1
    assert payload["memory"]["peak_rss_bytes"] > 0
    assert "initial_accuracy=" in completed.stderr
