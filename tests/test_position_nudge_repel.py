"""Slice 2 tests: position_nudge_repel preserves pre-nudge x/y."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ggrepel_py.position_nudge_repel import (
    PositionNudgeRepel,
    position_nudge_repel,
)


def _df(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "x": np.linspace(0, 1, n),
        "y": np.linspace(0, 2, n),
        "label": [f"p{i}" for i in range(n)],
    })


def test_factory_returns_positionnudgerepel():
    p = position_nudge_repel(x=0.1, y=0.2)
    assert isinstance(p, PositionNudgeRepel)
    assert p.x == 0.1
    assert p.y == 0.2


def test_compute_layer_shifts_and_preserves_orig():
    df = _df()
    p = position_nudge_repel(x=1.0, y=-0.5)
    params = p.setup_params(df)
    out = p.compute_layer(df, params)
    np.testing.assert_allclose(out["x"].to_numpy(),
                               df["x"].to_numpy() + 1.0)
    np.testing.assert_allclose(out["y"].to_numpy(),
                               df["y"].to_numpy() - 0.5)
    np.testing.assert_array_equal(out["x_orig"].to_numpy(), df["x"].to_numpy())
    np.testing.assert_array_equal(out["y_orig"].to_numpy(), df["y"].to_numpy())


def test_compute_layer_zero_nudge_still_preserves_orig():
    df = _df()
    p = position_nudge_repel()
    params = p.setup_params(df)
    out = p.compute_layer(df, params)
    np.testing.assert_array_equal(out["x"].to_numpy(), df["x"].to_numpy())
    np.testing.assert_array_equal(out["y"].to_numpy(), df["y"].to_numpy())
    assert "x_orig" in out.columns
    assert "y_orig" in out.columns


def test_compute_layer_x_only():
    df = _df()
    p = position_nudge_repel(x=0.2)
    out = p.compute_layer(df, p.setup_params(df))
    np.testing.assert_allclose(out["x"].to_numpy(),
                               df["x"].to_numpy() + 0.2)
    np.testing.assert_array_equal(out["y"].to_numpy(), df["y"].to_numpy())
    np.testing.assert_array_equal(out["x_orig"].to_numpy(), df["x"].to_numpy())
