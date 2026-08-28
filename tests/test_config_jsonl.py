from __future__ import annotations

import json
import math
import os
from pathlib import Path

from affine_hints.config import append_jsonl, read_jsonl


def test_jsonl_writer_replaces_nonfinite_values_with_null() -> None:
    path = Path.cwd() / f".pytest-jsonl-{os.getpid()}.jsonl"
    try:
        append_jsonl(path, [{"finite": 1.0, "nan": math.nan, "positive_inf": math.inf}])
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw == {"finite": 1.0, "nan": None, "positive_inf": None}
        assert read_jsonl(path) == [raw]
    finally:
        path.unlink(missing_ok=True)
