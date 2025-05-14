import numpy as np


def safe_allclose(a: list[float | None], b: list[float | None], rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    a_filtered = [x for x in a if x is not None]
    b_filtered = [x for x in b if x is not None]

    if len(a_filtered) != len(b_filtered):
        return False

    if not a_filtered:
        return True

    return bool(np.allclose(np.array(a_filtered), np.array(b_filtered), rtol=rtol, atol=atol))
