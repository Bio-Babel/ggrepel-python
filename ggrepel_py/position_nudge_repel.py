"""``position_nudge_repel`` and its ``PositionNudgeRepel`` class.

Port of ``ggrepel/R/position-nudge-repel.R``. The distinguishing feature over
``ggplot2_py.PositionNudge`` is that we preserve the pre-nudge ``x``/``y`` as
``x_orig``/``y_orig`` columns — the repel geoms use those to anchor the leader
line for each label.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from ggplot2_py.position import PositionNudge, _transform_position

__all__ = ["PositionNudgeRepel", "position_nudge_repel"]


class PositionNudgeRepel(PositionNudge):
    """``PositionNudge`` variant that remembers the pre-nudge coordinates.

    Sets ``x_orig`` and ``y_orig`` on the output DataFrame before applying the
    nudge. The repel geoms draw the leader line from ``(x_orig, y_orig)`` to
    the nudged ``(x, y)``.
    """

    x: Any = 0
    y: Any = 0

    def __init__(self, x: Any = 0, y: Any = 0, **kwargs: Any) -> None:
        super().__init__(x=x, y=y, **kwargs)

    def setup_params(self, data: pd.DataFrame) -> Dict[str, Any]:
        return {"x": self.x, "y": self.y}

    def compute_layer(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any],
        layout: Any = None,
    ) -> pd.DataFrame:
        x_orig = data["x"].to_numpy() if "x" in data.columns else None
        y_orig = data["y"].to_numpy() if "y" in data.columns else None

        px = params.get("x", 0)
        py = params.get("y", 0)
        has_nonzero_x = np.any(np.asarray(px) != 0)
        has_nonzero_y = np.any(np.asarray(py) != 0)

        if has_nonzero_x and has_nonzero_y:
            data = _transform_position(data, lambda v: v + px, lambda v: v + py)
        elif has_nonzero_x:
            data = _transform_position(data, lambda v: v + px, None)
        elif has_nonzero_y:
            data = _transform_position(data, None, lambda v: v + py)
        else:
            data = data.copy()

        if x_orig is not None:
            data["x_orig"] = x_orig
        if y_orig is not None:
            data["y_orig"] = y_orig
        return data


def position_nudge_repel(x: Any = 0, y: Any = 0) -> PositionNudgeRepel:
    """Nudge labels by a constant offset, preserving the original ``x``/``y``.

    Parameters
    ----------
    x, y
        Horizontal / vertical shift applied to the data before repulsion.

    Returns
    -------
    PositionNudgeRepel
    """
    return PositionNudgeRepel(x=x, y=y)
