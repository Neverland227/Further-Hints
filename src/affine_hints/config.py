"""Configuration, manifests, checkpoints, and bounded-run helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


class ConfigurationError(ValueError):
    """Raised when an experiment configuration violates a hard boundary."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and retain its source path for manifests."""

    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"configuration must be a mapping: {source}")
    data["_config_path"] = str(source)
    return data


def stable_hash(value: Any) -> str:
    """Return a stable SHA-256 hash for JSON-compatible configuration data."""

    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    """Atomically replace a UTF-8 text file in its destination directory."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Append complete JSON records and flush them for checkpoint safety."""

    materialized = list(rows)
    if not materialized:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        for row in materialized:
            handle.write(json.dumps(_json_safe(row), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL records, tolerating only a torn final line."""

    source = Path(path)
    if not source.exists():
        return []
    lines = source.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
    return rows


def atomic_write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Atomically replace a complete JSONL snapshot."""

    payload = "".join(
        json.dumps(_json_safe(row), sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    )
    atomic_write_text(path, payload)


def _json_safe(value: Any) -> Any:
    """Convert nested records to strict JSON, mapping non-finite values to null."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _git_state(root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def _source_tree_hash(root: Path) -> str:
    """Hash the runnable source/configuration tree, excluding generated data."""

    digest = hashlib.sha256()
    excluded_parts = {"results", ".venv", "__pycache__", ".pytest_cache", "calibration"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in excluded_parts for part in path.relative_to(root).parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("numpy", "scipy", "sympy", "pandas", "matplotlib", "yaml", "networkx", "psutil", "fpylll"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[name] = None
    return versions


@dataclass
class RunContext:
    """Filesystem and timing state for one reproducible experiment run."""

    phase: str
    run_id: str
    root: Path
    run_dir: Path
    config: dict[str, Any]
    start_monotonic: float
    is_resume: bool = False

    @classmethod
    def create(cls, phase: str, config: dict[str, Any], project_root: str | Path) -> "RunContext":
        root = Path(project_root).resolve()
        resume_dir_value = config.get("_resume_dir")
        if resume_dir_value:
            run_dir = Path(str(resume_dir_value)).resolve()
            expected_parent = (root / "results" / phase).resolve()
            if run_dir.parent != expected_parent:
                raise ConfigurationError(f"resume directory is not a {phase} run: {run_dir}")
            recorded_path = run_dir / "config.yaml"
            if not recorded_path.exists():
                raise ConfigurationError(f"resume directory has no config.yaml: {run_dir}")
            recorded = yaml.safe_load(recorded_path.read_text(encoding="utf-8")) or {}
            requested = {k: v for k, v in config.items() if not k.startswith("_")}
            if stable_hash(recorded) != stable_hash(requested):
                raise ConfigurationError("resume configuration differs from the recorded run configuration")
            return cls(phase, run_dir.name, root, run_dir, config, time.monotonic(), True)
        configured = config.get("run_id")
        if configured:
            run_id = str(configured)
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_id = f"{stamp}-{stable_hash({k: v for k, v in config.items() if not k.startswith('_')})[:8]}"
        run_dir = root / "results" / phase / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "figures").mkdir(exist_ok=True)
        config_copy = {k: v for k, v in config.items() if not k.startswith("_")}
        atomic_write_text(run_dir / "config.yaml", yaml.safe_dump(config_copy, sort_keys=False))
        environment = {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "cpu": platform.processor() or platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "ram_bytes": _ram_bytes(),
            "python_version": sys.version,
            "dependency_versions": _dependency_versions(),
        }
        atomic_write_text(run_dir / "environment.json", json.dumps(environment, indent=2, sort_keys=True) + "\n")
        for filename in ("trials.jsonl", "failures.jsonl"):
            target = run_dir / filename
            if not target.exists():
                atomic_write_text(target, "")
        return cls(phase, run_id, root, run_dir, config, time.monotonic(), False)

    def manifest(self, *, status: str, peak_rss_bytes: int | None = None, derived_seeds: Any = None) -> dict[str, Any]:
        commit, dirty = _git_state(self.root)
        return {
            "schema_version": 1,
            "phase": self.phase,
            "run_id": self.run_id,
            "status": status,
            "git_commit": commit,
            "dirty_state": dirty,
            "source_tree_sha256": _source_tree_hash(self.root),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "cpu": platform.processor() or platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "ram_bytes": _ram_bytes(),
            "python_version": sys.version,
            "dependency_versions": _dependency_versions(),
            "master_seed": self.config.get("master_seed"),
            "derived_seeds": derived_seeds,
            "config_hash": stable_hash({k: v for k, v in self.config.items() if not k.startswith("_")}),
            "command_line": sys.argv,
            "wall_clock_seconds": time.monotonic() - self.start_monotonic,
            "peak_rss_bytes": peak_rss_bytes,
        }

    def write_manifest(self, **kwargs: Any) -> None:
        if kwargs.get("peak_rss_bytes") is None:
            kwargs["peak_rss_bytes"] = process_peak_rss_bytes()
        atomic_write_text(
            self.run_dir / "manifest.json",
            json.dumps(self.manifest(**kwargs), indent=2, sort_keys=True, default=_json_default) + "\n",
        )


def _ram_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except ImportError:
        return None


def process_peak_rss_bytes() -> int | None:
    """Best available process peak RSS; current RSS is a Windows fallback."""

    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB, macOS bytes.
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, OSError):
        try:
            import psutil

            return int(psutil.Process().memory_info().rss)
        except (ImportError, OSError):
            return None


def require_common_limits(config: dict[str, Any]) -> None:
    """Fail closed unless all cross-phase resource and seed limits are explicit."""

    required = (
        "master_seed",
        "max_wall_time",
        "max_memory",
        "max_candidates",
        "max_exact_states",
        "beta_max",
        "enumeration_node_limit",
        "calibration_seeds",
        "confirmation_seeds",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ConfigurationError(f"missing required configuration fields: {missing}")
    if int(config["max_candidates"]) <= 0 or int(config["max_exact_states"]) <= 0:
        raise ConfigurationError("candidate and exact-state limits must be positive")
    if int(config["beta_max"]) > 30:
        raise ConfigurationError("bounded real-lattice beta_max must not exceed 30")


def validate_real_lattice_bounds(config: dict[str, Any]) -> None:
    """Enforce the task's toy-only real-lattice boundary."""

    require_common_limits(config)
    dimensions = [int(v) for v in config.get("n", [])]
    dimensions.extend(int(regime["n"]) for regime in config.get("regimes", []) if "n" in regime)
    if any(v > 80 for v in dimensions):
        raise ConfigurationError("real-lattice dimensions are capped at n=80")
    grid_betas = [int(value) for value in config.get("grid", {}).get("beta", [])]
    if any(value > int(config["beta_max"]) for value in grid_betas):
        raise ConfigurationError("a finite-grid beta exceeds beta_max")
    if config.get("g6k", False):
        raise ConfigurationError("G6K is DEFERRED_OUT_OF_SCOPE")
