"""Memory-limit tests for sparse benchmark supervision."""

import importlib.util
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

MODULE_PATH = pathlib.Path(__file__).with_name("sparse_benchmark_supervisor.py")


def _load():
    spec = importlib.util.spec_from_file_location("_sparse_supervisor_memory", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeProcess:
    def __init__(self, rss_bytes=0, descendants=()):
        self.rss_bytes = rss_bytes
        self.descendants = list(descendants)

    def children(self, recursive=False):
        return self.descendants

    def memory_info(self):
        return SimpleNamespace(rss=self.rss_bytes)


def test_memory_guard_reason_uses_strict_boundaries():
    supervisor = _load()
    limits = supervisor.ResourceLimits(100, 50)

    assert (
        supervisor.memory_guard_reason(supervisor.MemorySample(100, 50), limits)
        is None
    )
    assert supervisor.memory_guard_reason(supervisor.MemorySample(101, 50), limits) == (
        "rss_limit_exceeded"
    )
    assert supervisor.memory_guard_reason(supervisor.MemorySample(100, 49), limits) == (
        "available_memory_below_minimum"
    )


def test_sample_memory_sums_root_and_descendants(monkeypatch):
    supervisor = _load()
    root = _FakeProcess(10, [_FakeProcess(20)])
    monkeypatch.setattr(
        supervisor.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=70),
    )

    assert supervisor._sample_memory(root) == supervisor.MemorySample(30, 70)


def test_run_supervised_stops_at_failed_preflight(monkeypatch):
    supervisor = _load()
    monkeypatch.setattr(
        supervisor.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=49),
    )
    monkeypatch.setattr(
        supervisor.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("launched")),
    )

    result = supervisor.run_supervised(
        ["worker"], supervisor.ResourceLimits(100, 50)
    )

    assert result == supervisor.SupervisedResult(
        2, "", 0, "memory_guard", "available_memory_below_minimum"
    )


@pytest.mark.parametrize("exit_code,status", [(0, "completed"), (7, "failed")])
def test_run_supervised_preserves_output_and_exit_status(exit_code, status):
    supervisor = _load()
    code = (
        "import sys; print('out', flush=True); "
        "print('err', file=sys.stderr, flush=True); "
        f"raise SystemExit({exit_code})"
    )
    result = supervisor.run_supervised(
        [sys.executable, "-c", code], supervisor.ResourceLimits(2**62, 0)
    )

    assert result.exit_code == exit_code
    assert result.status == status
    assert result.stdout.splitlines() == ["out", "err"]
