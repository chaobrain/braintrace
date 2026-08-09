"""Process-lifecycle tests for sparse benchmark supervision."""

import importlib.util
import pathlib
import subprocess
import sys
from types import SimpleNamespace

MODULE_PATH = pathlib.Path(__file__).with_name("sparse_benchmark_supervisor.py")


def _load():
    spec = importlib.util.spec_from_file_location("_sparse_supervisor_process", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeProcess:
    def __init__(self, descendants=()):
        self.descendants = list(descendants)
        self.terminate_count = 0
        self.kill_count = 0

    def children(self, recursive=False):
        return self.descendants

    def terminate(self):
        self.terminate_count += 1

    def kill(self):
        self.kill_count += 1


class _FakeChild:
    pid = 17

    def __init__(self):
        self.wait_count = 0

    def poll(self):
        return None

    def wait(self, timeout=None):
        self.wait_count += 1
        return 0

    def kill(self):
        return None


def test_run_supervised_terminates_then_kills_guard_survivors(monkeypatch):
    supervisor = _load()
    child = _FakeChild()
    descendant = _FakeProcess()
    root = _FakeProcess([descendant])

    def launch(command, **options):
        options["stdout"].write("partial output\n")
        options["stdout"].flush()
        assert options["stderr"] is subprocess.STDOUT
        return child

    monkeypatch.setattr(supervisor.subprocess, "Popen", launch)
    monkeypatch.setattr(supervisor.psutil, "Process", lambda pid: root)
    monkeypatch.setattr(
        supervisor.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=1_000),
    )
    monkeypatch.setattr(
        supervisor,
        "_sample_memory",
        lambda process: supervisor.MemorySample(101, 1_000),
    )
    wait_calls = []

    def wait_procs(processes, timeout):
        wait_calls.append(list(processes))
        return ([], [root]) if len(wait_calls) == 1 else (list(processes), [])

    monkeypatch.setattr(supervisor.psutil, "wait_procs", wait_procs)
    result = supervisor.run_supervised(
        ["worker"], supervisor.ResourceLimits(100, 50)
    )

    assert result.status == "memory_guard"
    assert result.guard_reason == "rss_limit_exceeded"
    assert descendant.terminate_count == root.terminate_count == 1
    assert root.kill_count == 1
    assert len(wait_calls) == 2
    assert child.wait_count == 1
