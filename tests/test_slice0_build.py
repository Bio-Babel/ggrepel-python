"""Slice 0 smoke tests: _repel extension builds and load_mtcars works."""

from __future__ import annotations

import numpy as np
import pytest

from ggrepel_py import _repel
from ggrepel_py.data import load_mtcars


def test_euclid_simple():
    assert _repel.euclid(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == pytest.approx(5.0)


def test_centroid_middle():
    out = _repel.centroid(np.array([0.0, 0.0, 2.0, 4.0]), 0.5, 0.5)
    assert list(out) == [1.0, 2.0]


def test_approximately_equal():
    assert _repel.approximately_equal(1.0, 1.0 + 1e-14)
    assert not _repel.approximately_equal(1.0, 1.01)


def test_repel_boxes2_smoke():
    rng = np.random.default_rng(0)
    n = 3
    pts = rng.random((n, 2)) * 10
    boxes = np.column_stack([pts[:, 0] - 0.5, pts[:, 1] - 0.25,
                             pts[:, 0] + 0.5, pts[:, 1] + 0.25])
    out = _repel.repel_boxes2(
        data_points=pts,
        point_size=np.zeros(n),
        point_padding_x=0.01, point_padding_y=0.01,
        boxes=boxes,
        xlim=np.array([0.0, 10.0]),
        ylim=np.array([0.0, 10.0]),
        hjust=np.full(n, 0.5),
        vjust=np.full(n, 0.5),
        force_push=1e-6, force_pull=1e-6,
        max_time=1.0, max_overlaps=10.0, max_iter=200,
        direction="both", verbose=0, seed=42,
    )
    assert set(out.keys()) == {"x", "y", "too_many_overlaps"}
    assert out["x"].shape == (n,)
    assert out["y"].shape == (n,)
    assert out["too_many_overlaps"].dtype == np.bool_


def test_repel_boxes2_is_deterministic_under_same_seed():
    rng = np.random.default_rng(1)
    n = 5
    pts = rng.random((n, 2)) * 10
    boxes = np.column_stack([pts[:, 0] - 0.5, pts[:, 1] - 0.25,
                             pts[:, 0] + 0.5, pts[:, 1] + 0.25])
    kwargs = dict(
        data_points=pts, point_size=np.zeros(n),
        point_padding_x=0.01, point_padding_y=0.01,
        boxes=boxes,
        xlim=np.array([0.0, 10.0]), ylim=np.array([0.0, 10.0]),
        hjust=np.full(n, 0.5), vjust=np.full(n, 0.5),
        force_push=1e-6, force_pull=1e-6,
        max_time=1.0, max_overlaps=10.0, max_iter=500,
        direction="both", verbose=0, seed=123,
    )
    a = _repel.repel_boxes2(**kwargs)
    b = _repel.repel_boxes2(**kwargs)
    np.testing.assert_array_equal(a["x"], b["x"])
    np.testing.assert_array_equal(a["y"], b["y"])


def test_load_mtcars_shape():
    df = load_mtcars()
    assert df.shape == (32, 12)
    assert df.columns[0] == "car"
    for col in ["mpg", "cyl", "disp", "hp", "drat", "wt", "qsec", "vs", "am", "gear", "carb"]:
        assert col in df.columns
