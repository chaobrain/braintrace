"""Supervise a sparse benchmark worker under host-memory limits."""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, Sequence, TextIO

import psutil

_MEMORY_GUARD_EXIT_CODE = 2
_TIMEOUT_EXIT_CODE = 124
_POLL_INTERVAL_SECONDS = 0.1
_PROCESS_WAIT_SECONDS = 2.0
_PROCESS_ERRORS = (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess)


@dataclass(frozen=True)
class ResourceLimits:
    """Host-memory limits applied to a supervised benchmark process.

    Attributes
    ----------
    max_rss_bytes : int
        Maximum permitted resident memory for the worker process tree.
    min_available_bytes : int
        Minimum host-available memory that must remain.
    max_wall_seconds : float
        Maximum permitted worker wall-clock duration.
    """

    max_rss_bytes: int
    min_available_bytes: int
    max_wall_seconds: float = 1800.0


@dataclass(frozen=True)
class MemorySample:
    """Process-tree RSS and host-available memory observed together.

    Attributes
    ----------
    rss_bytes : int
        Sum of resident memory across the worker process tree.
    available_bytes : int
        Host-available memory observed with the process sample.
    """

    rss_bytes: int
    available_bytes: int


@dataclass(frozen=True)
class SupervisedResult:
    """Captured outcome of a supervised benchmark process.

    Attributes
    ----------
    exit_code : int
        Worker exit code, or the supervisor memory-guard code.
    stdout : str
        Combined standard output and standard error from the worker.
    peak_rss_bytes : int
        Highest sampled process-tree resident memory.
    status : str
        One of ``completed``, ``failed``, or ``memory_guard``.
    guard_reason : str or None
        Stable memory-guard identifier when a limit stopped launch or execution.
    """

    exit_code: int
    stdout: str
    peak_rss_bytes: int
    status: str
    guard_reason: str | None


def memory_guard_reason(
    sample: MemorySample, limits: ResourceLimits
) -> str | None:
    """Return the memory-limit violation represented by ``sample``.

    Parameters
    ----------
    sample : MemorySample
        Process and host memory observed at one instant.
    limits : ResourceLimits
        Maximum process-tree RSS and minimum host memory headroom.

    Returns
    -------
    str or None
        Stable violation identifier, or ``None`` when the sample is safe.
    """
    if sample.rss_bytes > limits.max_rss_bytes:
        return "rss_limit_exceeded"
    if sample.available_bytes < limits.min_available_bytes:
        return "available_memory_below_minimum"
    return None


def _processes_in_tree(root: psutil.Process) -> list[psutil.Process]:
    try:
        descendants = root.children(recursive=True)
    except _PROCESS_ERRORS:
        descendants = []
    return [*descendants, root]


def _process_rss(process: psutil.Process) -> int:
    try:
        return int(process.memory_info().rss)
    except _PROCESS_ERRORS:
        return 0


def _sample_memory(root: psutil.Process) -> MemorySample:
    rss_bytes = sum(_process_rss(process) for process in _processes_in_tree(root))
    available_bytes = int(psutil.virtual_memory().available)
    return MemorySample(rss_bytes, available_bytes)


def _attempt(operation: Callable[[], object]) -> None:
    try:
        operation()
    except _PROCESS_ERRORS:
        return


def _wait_for_processes(processes: list[psutil.Process]) -> list[psutil.Process]:
    try:
        return psutil.wait_procs(processes, timeout=_PROCESS_WAIT_SECONDS)[1]
    except psutil.Error:
        return processes


def _stop_process_tree(child: subprocess.Popen[str], root: psutil.Process) -> None:
    processes = _processes_in_tree(root)
    for process in processes:
        _attempt(process.terminate)
    survivors = _wait_for_processes(processes)
    for process in survivors:
        _attempt(process.kill)
    if survivors:
        _wait_for_processes(survivors)
    try:
        child.wait(timeout=_PROCESS_WAIT_SECONDS)
    except subprocess.TimeoutExpired:
        child.kill()
        try:
            child.wait(timeout=_PROCESS_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            return


def _subprocess_isolation() -> dict[str, object]:
    if os.name == "nt":
        return {
            "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP"),
        }
    return {"start_new_session": True}


def _read_output(output: TextIO) -> str:
    output.seek(0)
    return str(output.read())


def _preflight(limits: ResourceLimits) -> str | None:
    sample = MemorySample(0, int(psutil.virtual_memory().available))
    return memory_guard_reason(sample, limits)


def run_supervised(
    command: Sequence[str], limits: ResourceLimits
) -> SupervisedResult:
    """Run one isolated worker while enforcing process-tree memory limits.

    Parameters
    ----------
    command : sequence of str
        Executable and arguments passed directly to ``subprocess.Popen``.
    limits : ResourceLimits
        Host-memory limits checked before launch and while the worker runs.

    Returns
    -------
    SupervisedResult
        Captured combined output, peak RSS, exit status, and guard reason.
    """
    preflight_reason = _preflight(limits)
    if preflight_reason is not None:
        return SupervisedResult(
            _MEMORY_GUARD_EXIT_CODE, "", 0, "memory_guard", preflight_reason
        )
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as output:
        try:
            child = subprocess.Popen(
                list(command),
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                close_fds=True,
                **_subprocess_isolation(),
            )
        except OSError as error:
            return SupervisedResult(1, str(error), 0, "failed", None)
        try:
            root = psutil.Process(child.pid)
        except psutil.NoSuchProcess:
            exit_code = child.wait()
            status = "completed" if exit_code == 0 else "failed"
            return SupervisedResult(exit_code, _read_output(output), 0, status, None)
        peak_rss_bytes = 0
        guard_reason = None
        timed_out = False
        started = time.monotonic()
        try:
            while child.poll() is None:
                sample = _sample_memory(root)
                peak_rss_bytes = max(peak_rss_bytes, sample.rss_bytes)
                guard_reason = memory_guard_reason(sample, limits)
                if guard_reason is not None:
                    _stop_process_tree(child, root)
                    break
                if time.monotonic() - started > limits.max_wall_seconds:
                    guard_reason = "wall_time_exceeded"
                    timed_out = True
                    _stop_process_tree(child, root)
                    break
                time.sleep(_POLL_INTERVAL_SECONDS)
        except BaseException:
            _stop_process_tree(child, root)
            raise
        if timed_out:
            exit_code = _TIMEOUT_EXIT_CODE
            status = "timeout"
        elif guard_reason is not None:
            exit_code = _MEMORY_GUARD_EXIT_CODE
            status = "memory_guard"
        else:
            final_sample = _sample_memory(root)
            peak_rss_bytes = max(peak_rss_bytes, final_sample.rss_bytes)
            exit_code = child.wait()
            status = "completed" if exit_code == 0 else "failed"
        return SupervisedResult(
            exit_code,
            _read_output(output),
            peak_rss_bytes,
            status,
            guard_reason,
        )
