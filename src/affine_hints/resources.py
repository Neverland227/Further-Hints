"""Short-lived worker pools and external-process resource guards."""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import subprocess
import time
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def parse_bytes(value: int | float | str) -> int:
    """Parse a positive byte count such as ``8GB`` or ``512MiB``."""

    if isinstance(value, (int, float)):
        return int(value)
    text = value.strip().upper().replace("IB", "B")
    multipliers = {"KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4, "B": 1}
    for suffix in ("TB", "GB", "MB", "KB", "B"):
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * multipliers[suffix])
    return int(text)


def apply_unix_task_limits(*, max_wall_seconds: int, max_address_space: int | float | str) -> None:
    """Apply per-process hard limits on Linux; no-op on unsupported systems."""

    try:
        import resource
    except ImportError:
        return
    address_bytes = parse_bytes(max_address_space)
    current_soft, current_hard = resource.getrlimit(resource.RLIMIT_AS)
    hard = address_bytes if current_hard in (-1, resource.RLIM_INFINITY) else min(address_bytes, current_hard)
    resource.setrlimit(resource.RLIMIT_AS, (min(address_bytes, hard), hard))

    def timeout_handler(signum, frame):  # noqa: ANN001
        del signum, frame
        raise TimeoutError("RESOURCE_LIMIT: per-task max_wall_time exceeded")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(max(1, int(max_wall_seconds)))


def _set_single_thread_environment() -> None:
    """Set thread caps before spawn so child imports see them immediately."""

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _worker_initializer() -> None:
    # Keep the caps in place if a worker or native library mutates its
    # environment after process start.
    _set_single_thread_environment()


def bounded_pool_map(
    function: Callable[[T], R],
    tasks: Iterable[T],
    *,
    workers: int,
    max_tasks_per_child: int,
    chunksize: int = 1,
) -> list[R]:
    """Spawn-isolated pool whose workers retire after a bounded task count."""

    if workers <= 1:
        return [function(task) for task in tasks]
    # With the spawn start method, the child imports the experiment module
    # before Pool.initializer runs. Set these in the parent first so NumPy,
    # OpenBLAS, MKL, and NumExpr cannot each create a full 64-thread team while
    # the child is importing.
    _set_single_thread_environment()
    context = mp.get_context("spawn")
    with context.Pool(
        processes=workers,
        initializer=_worker_initializer,
        maxtasksperchild=max_tasks_per_child,
    ) as pool:
        return pool.map(function, tasks, chunksize=chunksize)


def bounded_pool_imap_unordered(
    function: Callable[[T], R],
    tasks: Iterable[T],
    *,
    workers: int,
    max_tasks_per_child: int,
    chunksize: int = 1,
):
    """Yield completed tasks so the caller can checkpoint each result."""

    task_list = list(tasks)
    if workers <= 1:
        for task in task_list:
            yield function(task)
        return
    _set_single_thread_environment()
    context = mp.get_context("spawn")
    with context.Pool(
        processes=workers,
        initializer=_worker_initializer,
        maxtasksperchild=max_tasks_per_child,
    ) as pool:
        yield from pool.imap_unordered(function, task_list, chunksize=chunksize)


def run_short_lived_command(
    command: list[str],
    *,
    timeout_seconds: float,
    max_rss_bytes: int,
    poll_seconds: float = 0.25,
) -> dict[str, Any]:
    """Run one external job, killing its process tree on timeout/RSS breach."""

    import psutil

    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = "1"
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment)
    ps_process = psutil.Process(process.pid)
    peak = 0
    reason: str | None = None
    while process.poll() is None:
        elapsed = time.monotonic() - started
        try:
            family = [ps_process] + ps_process.children(recursive=True)
            rss = sum(child.memory_info().rss for child in family if child.is_running())
            peak = max(peak, rss)
        except psutil.Error:
            rss = 0
        if elapsed > timeout_seconds:
            reason = "max_wall_time"
        elif max_rss_bytes and rss > max_rss_bytes:
            reason = "max_RSS"
        if reason:
            try:
                for child in ps_process.children(recursive=True):
                    child.kill()
                ps_process.kill()
            except psutil.Error:
                pass
            break
        time.sleep(poll_seconds)
    stdout, stderr = process.communicate()
    return {
        "returncode": process.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "wall_time": time.monotonic() - started,
        "peak_rss_bytes": peak,
        "status": "RESOURCE_LIMIT" if reason else ("COMPLETED" if process.returncode == 0 else "FAILED"),
        "resource_limit": reason,
    }
