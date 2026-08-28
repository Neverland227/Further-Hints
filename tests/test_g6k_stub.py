from __future__ import annotations

import numpy as np
import pytest

from affine_hints.candidates.g6k_backend import DeferredOutOfScope, G6KSieveSource


def test_g6k_is_permanently_disabled() -> None:
    with pytest.raises(DeferredOutOfScope, match="DEFERRED_OUT_OF_SCOPE"):
        G6KSieveSource().generate(1, np.random.default_rng(1))

