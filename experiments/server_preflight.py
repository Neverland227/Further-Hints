"""Read-only preflight for the AMD EPYC Turin experiment server."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import psutil
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "server_turin.yaml"))
    arguments = parser.parse_args()
    config = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    required = ("numpy", "scipy", "sympy", "pandas", "matplotlib", "yaml", "networkx", "pytest", "psutil", "fpylll")
    result = {
        "platform": platform.platform(),
        "python": sys.version,
        "logical_cpus": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "dependencies": {name: bool(importlib.util.find_spec(name)) for name in required},
        "configured_pools": {
            "synthetic_workers": config["synthetic_pool"]["workers"],
            "lattice_workers": config["lattice_pool"]["workers"],
            "sage_enabled": config["sage_pool"]["enabled"],
        },
        "checks": {},
    }
    visible_ram_gb = psutil.virtual_memory().total / 1024**3
    reserve_gb = float(config["hardware"]["reserved_ram_gb"])
    synthetic_budget_gb = (
        float(config["synthetic_pool"]["workers"])
        * float(config["synthetic_pool"]["estimated_rss_gb_per_worker"])
        + reserve_gb
    )
    lattice_rss_text = str(config["lattice_pool"]["max_RSS_per_worker"]).upper().replace("GIB", "").replace("GB", "")
    lattice_budget_gb = float(config["lattice_pool"]["workers"]) * float(lattice_rss_text) + reserve_gb
    result["checks"]["python_3_11_or_newer"] = sys.version_info >= (3, 11)
    result["checks"]["at_least_64_logical_cpus"] = (os.cpu_count() or 0) >= 64
    result["checks"]["at_least_120_gib_visible"] = psutil.virtual_memory().total >= 120 * 1024**3
    result["checks"]["core_dependencies"] = all(result["dependencies"][name] for name in required if name != "fpylll")
    result["checks"]["optional_lattice_dependency"] = result["dependencies"]["fpylll"]
    result["checks"]["sage_not_required"] = True
    result["checks"]["worker_count_fits_visible_cpus"] = int(config["synthetic_pool"]["workers"]) <= (os.cpu_count() or 0)
    result["checks"]["synthetic_memory_budget_fits"] = synthetic_budget_gb <= visible_ram_gb
    result["checks"]["lattice_memory_budget_fits"] = lattice_budget_gb <= visible_ram_gb
    result["checks"]["short_lived_lattice_workers"] = int(config["lattice_pool"]["max_tasks_per_child"]) == 1
    result["checks"]["sage_disabled"] = not bool(config["sage_pool"]["enabled"])
    result["checks"]["one_blas_thread_per_worker"] = int(config["threading"]["blas_threads_per_worker"]) == 1
    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        git_checkout_detected = True
    except (OSError, subprocess.SubprocessError):
        git_checkout_detected = False
    result["checks"]["git_checkout_detected"] = git_checkout_detected
    result["configured_memory_budget_gb"] = {
        "synthetic_including_reserve": synthetic_budget_gb,
        "lattice_including_reserve": lattice_budget_gb,
    }
    result["ready_for_core"] = all(
        result["checks"][key]
        for key in (
            "python_3_11_or_newer",
            "at_least_64_logical_cpus",
            "at_least_120_gib_visible",
            "core_dependencies",
            "worker_count_fits_visible_cpus",
            "synthetic_memory_budget_fits",
            "sage_disabled",
            "one_blas_thread_per_worker",
            "git_checkout_detected",
        )
    )
    result["ready_for_phase2"] = all(
        (
            result["ready_for_core"],
            result["checks"]["optional_lattice_dependency"],
            result["checks"]["lattice_memory_budget_fits"],
            result["checks"]["short_lived_lattice_workers"],
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
