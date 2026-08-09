"""Tests for configurable sparse pp-prop benchmark orchestration."""

import json

from sparse_benchmark_config import SparseBenchmarkConfig
from sparse_benchmark_supervisor import SupervisedResult

from configurable_sparse_benchmark import (
    _completed_payload,
    _exit_code,
    _failure_payload,
    _parse_worker_payload,
)


def _valid_payload(config: SparseBenchmarkConfig) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "completed",
        "config": __import__("sparse_benchmark_config").config_to_dict(config),
        "metrics": {"threshold_updates": None},
        "timings": {},
        "environment": {},
    }


def test_worker_payload_is_separated_from_progress() -> None:
    payload = _valid_payload(SparseBenchmarkConfig())
    output = f"update=1 loss=0.5\n{json.dumps(payload)}\n"

    parsed, progress = _parse_worker_payload(output)

    assert parsed == payload
    assert progress == ["update=1 loss=0.5"]


def test_completed_payload_adds_peak_memory() -> None:
    config = SparseBenchmarkConfig()
    output = json.dumps(_valid_payload(config))
    result = SupervisedResult(0, output, 2**30, "completed", None)

    payload = _completed_payload(config, result)

    assert payload["memory"]["peak_rss_bytes"] == 2**30
    assert payload["memory"]["peak_rss_gib"] == 1.0


def test_missing_schema_fields_fail_closed() -> None:
    config = SparseBenchmarkConfig()
    result = SupervisedResult(
        0, '{"schema_version": 1}', 10, "completed", None
    )

    payload = _completed_payload(config, result)

    assert payload["status"] == "failed"
    assert _exit_code(config, result, payload) == 1


def test_unknown_worker_status_is_rejected() -> None:
    payload = _valid_payload(SparseBenchmarkConfig())
    payload["status"] = "garbage"

    try:
        _parse_worker_payload(json.dumps(payload))
    except ValueError:
        return
    raise AssertionError("unknown status was accepted")


def test_memory_guard_failure_is_structured() -> None:
    config = SparseBenchmarkConfig()
    result = SupervisedResult(
        2, "", 512, "memory_guard", "rss_limit_exceeded"
    )

    payload = _failure_payload(config, result)

    assert payload["status"] == "memory_guard"
    assert payload["memory"]["guard_reason"] == "rss_limit_exceeded"


def test_require_target_returns_nonzero_for_a_miss() -> None:
    config = SparseBenchmarkConfig(require_target=True)
    result = SupervisedResult(0, "", 0, "completed", None)
    payload = {"metrics": {"threshold_updates": None}}

    assert _exit_code(config, result, payload) == 3
