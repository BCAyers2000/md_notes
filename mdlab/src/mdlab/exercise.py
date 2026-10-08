"""Checks for notebook exercises.

An exercise cell leaves ``answer = None`` for the reader to fill in. The
check reports 'not attempted' for None, so a notebook still runs top to
bottom unattempted, and fails loudly for a wrong answer.
"""

from __future__ import annotations

import numpy as np


def check(
    answer, target, rtol: float = 1e-3, atol: float = 0.0, name: str = "answer"
) -> bool | None:
    """Compare an exercise answer with its target.

    Parameters
    ----------
    answer : float, array or None
        The reader's result; None means not yet attempted.
    target : float or array
        The expected result, in the same units as ``answer``.
    rtol, atol : float
        Relative and absolute tolerance, as in :func:`numpy.allclose`.
    name : str
        Label used in the message.

    Returns
    -------
    bool or None
        True if correct, None if not attempted.

    Raises
    ------
    AssertionError
        If the answer has the wrong shape or is outside the tolerance.
    """
    if answer is None:
        print(f"{name}: not attempted yet")
        return None
    if np.shape(answer) != np.shape(target):
        raise AssertionError(
            f"{name} has shape {np.shape(answer)}, "
            f"but the target has shape {np.shape(target)}"
        )
    ok = np.allclose(answer, target, rtol=rtol, atol=atol)
    assert ok, f"{name} = {answer!r} is outside the tolerance of the target"
    print(f"{name}: correct")
    return True
