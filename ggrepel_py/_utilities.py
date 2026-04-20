"""Small utility helpers used across the repel geoms.

Mirrors ``ggrepel/R/utilities.R`` and the ``compute_just``/``just_dir`` helpers
that live in ``ggrepel/R/geom-text-repel.R``.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

import grid_py as grid

__all__ = [
    "PT",
    "compute_just",
    "exclude_outside",
    "ggname",
    "inside",
    "just_dir",
    "not_empty",
    "null_default",
    "to_unit",
]

# 72.27 points per inch, 25.4 mm per inch — ggplot2 font-size conversion constant.
PT: float = 72.27 / 25.4


def null_default(a: Any, b: Any) -> Any:
    """Port of the ``%||%`` operator: return ``a`` unless it is ``None``, else ``b``."""
    return a if a is not None else b


def ggname(prefix: str, grob: Any) -> Any:
    """Give a grob a predictable prefix-based name. Mirrors ``ggname()`` in ggplot2."""
    grob.name = grid.grob_name(grob, prefix=prefix)
    return grob


def to_unit(x: Any) -> Any:
    """Coerce ``x`` to a ``grid_py.Unit`` with ``"lines"`` default units."""
    if grid.is_unit(x):
        return x
    if x is None:
        return None
    # A scalar NaN is used to exclude points from repulsion calculations.
    if isinstance(x, float) and np.isnan(x):
        return float("nan")
    return grid.Unit(x, "lines")


def not_empty(xs: Sequence[Any]) -> np.ndarray:
    """Return a boolean mask indicating which items are non-empty / non-NA."""
    out = np.empty(len(xs), dtype=bool)
    for i, v in enumerate(xs):
        if v is None:
            out[i] = False
        elif isinstance(v, float) and np.isnan(v):
            out[i] = False
        else:
            out[i] = str(v) != ""
    return out


def inside(x: np.ndarray | Iterable[float], bounds: Sequence[float]) -> np.ndarray:
    """Return a boolean mask: points are inside the closed interval ``[lo, hi]``,
    with infinities passing through unchanged.
    """
    arr = np.asarray(x, dtype=float)
    lo, hi = bounds[0], bounds[1]
    return np.isinf(arr) | ((arr <= hi) & (arr >= lo))


def exclude_outside(data: pd.DataFrame, panel_scales: Any) -> pd.DataFrame:
    """Drop rows whose ``x``/``y`` fall outside the panel range.

    ``panel_scales`` mirrors R's panel params — a dict-like with either
    ``x.range``/``y.range`` (classic ggplot2) or ``x_range``/``y_range``
    (newer ``panel_params`` shape) entries.
    """

    def _get(name: str) -> Any:
        if isinstance(panel_scales, dict):
            return panel_scales.get(name)
        return getattr(panel_scales, name, None)

    xr = _get("x.range") or _get("x_range")
    yr = _get("y.range") or _get("y_range")
    if xr is None or yr is None:
        return data
    ix = inside(data["x"].to_numpy(), xr) & inside(data["y"].to_numpy(), yr)
    return data.loc[ix].reset_index(drop=True)


def just_dir(x: np.ndarray, tol: float = 0.001) -> np.ndarray:
    """Classify a vector into 1/2/3 bands (left/middle/right of 0.5)."""
    arr = np.asarray(x, dtype=float)
    out = np.full(arr.shape, 2, dtype=int)
    out[arr < 0.5 - tol] = 1
    out[arr > 0.5 + tol] = 3
    return out


_LEFT_MIDDLE_RIGHT = np.array(["left", "middle", "right"])
_RIGHT_MIDDLE_LEFT = np.array(["right", "middle", "left"])
_JUST_TO_NUMERIC = {
    "left": 0.0, "center": 0.5, "right": 1.0,
    "bottom": 0.0, "middle": 0.5, "top": 1.0,
}


def compute_just(
    just: np.ndarray | Sequence[str],
    a: np.ndarray | Sequence[float],
    b: np.ndarray | Sequence[float] | None = None,
    angle: np.ndarray | Sequence[float] | float = 0,
) -> np.ndarray:
    """Resolve ``"inward"``/``"outward"``/``"left"`` ... justification strings
    to numeric weights in ``[0, 1]``.

    Port of ``ggrepel:::compute_just``.
    """
    just_arr = np.asarray(just, dtype=object).astype(str)
    a_arr = np.asarray(a, dtype=float)
    b_arr = a_arr if b is None else np.asarray(b, dtype=float)
    angle_arr = np.broadcast_to(np.asarray(angle, dtype=float), just_arr.shape).copy()

    has_in_out = np.array([("inward" in j) or ("outward" in j) for j in just_arr])
    if has_in_out.any():
        angle_arr = angle_arr % 360.0
        angle_arr = np.where(angle_arr > 180, angle_arr - 360, angle_arr)
        angle_arr = np.where(angle_arr < -180, angle_arr + 360, angle_arr)

        rotated_forward = has_in_out & (angle_arr > 45) & (angle_arr < 135)
        rotated_backwards = has_in_out & (angle_arr < -45) & (angle_arr > -135)

        ab = np.where(rotated_forward | rotated_backwards, b_arr, a_arr)
        just_swap = rotated_backwards | (np.abs(angle_arr) > 135)
        inward = ((just_arr == "inward") & ~just_swap) | (
            (just_arr == "outward") & just_swap
        )
        outward = ((just_arr == "outward") & ~just_swap) | (
            (just_arr == "inward") & just_swap
        )

        if inward.any():
            dirs = just_dir(ab[inward])
            just_arr = just_arr.copy()
            just_arr[inward] = _LEFT_MIDDLE_RIGHT[dirs - 1]
        if outward.any():
            dirs = just_dir(ab[outward])
            just_arr[outward] = _RIGHT_MIDDLE_LEFT[dirs - 1]

    out = np.empty(just_arr.shape, dtype=float)
    for i, j in enumerate(just_arr):
        if j in _JUST_TO_NUMERIC:
            out[i] = _JUST_TO_NUMERIC[j]
        else:
            # numeric justification (already a number as string) — coerce.
            out[i] = float(j)
    return out
