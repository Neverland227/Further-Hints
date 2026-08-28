from __future__ import annotations

from affine_hints.structured.syndrome_mitm import syndrome_mitm


def test_syndrome_mitm_enforces_signed_fixed_weight_counts() -> None:
    matches = syndrome_mitm(
        H=((1, 1, 1, 1),),
        ell=(0,),
        q=17,
        support=(-1, 0, 1),
        max_states=100,
        max_matches=100,
        h_plus=1,
        h_minus=1,
    )
    assert matches
    assert all(row.count(1) == 1 and row.count(-1) == 1 for row in matches)
