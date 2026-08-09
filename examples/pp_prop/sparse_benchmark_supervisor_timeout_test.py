"""Timeout test for sparse benchmark process supervision."""

import importlib.util
import pathlib
import sys
from types import SimpleNamespace


MODULE_PATH = pathlib.Path(__file__).with_name("sparse_benchmark_supervisor.py")


def _load():
    spec = importlib.util.spec_from_file_location("_supervisor_timeout", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Child:
    pid = 17

    def poll(self):
        return None


class _Root:
    pass


def test_run_supervised_stops_worker_at_wall_limit(monkeypatch):
    supervisor = _load()
    child = _Child()
    root = _Root()
    stopped = []
    times = iter((0.0, 2.0))
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(supervisor.psutil, "Process", lambda pid: root)
    monkeypatch.setattr(
        supervisor.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=1_000),
    )
    monkeypatch.setattr(
        supervisor,
        "_sample_memory",
        lambda process: supervisor.MemorySample(10, 1_000),
    )
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(
        supervisor,
        "_stop_process_tree",
        lambda process, tree: stopped.append((process, tree)),
    )

    result = supervisor.run_supervised(
        ["worker"], supervisor.ResourceLimits(100, 50, 1.0)
    )

    assert result.exit_code == 124
    assert result.status == "timeout"
    assert result.guard_reason == "wall_time_exceeded"
    assert stopped == [(child, root)]
