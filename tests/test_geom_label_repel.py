"""Slice 4 tests: GeomLabelRepel, LabelRepelTree, and geom_label_repel factory."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import grid_py as grid
from grid_py import Unit

from ggrepel_py import (
    GeomLabelRepel,
    LabelRepelTree,
    geom_label_repel,
    position_nudge_repel,
)


# --------------------------------------------------------------------------- fixtures

def _tree_inputs(n: int = 5, seed: int = 42) -> dict:
    """Build a minimal kwarg dict suitable for ``LabelRepelTree.__init__``."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": np.linspace(0.1, 0.9, n) + rng.normal(0, 0.005, n),
        "y": np.linspace(0.1, 0.9, n) + rng.normal(0, 0.005, n),
        "label": [f"p{i}" for i in range(n)],
        "size": [3.88] * n,
        "angle": [0.0] * n,
        "hjust": [0.5] * n,
        "vjust": [0.5] * n,
        "colour": ["black"] * n,
        "fill": ["white"] * n,
        "alpha": [None] * n,
        "family": [""] * n,
        "fontface": [1] * n,
        "lineheight": [1.2] * n,
        "linewidth": [0.25] * n,
        "linetype": [1] * n,
        "point_size": [1.0] * n,
        "nudge_x": [0.0] * n,
        "nudge_y": [0.0] * n,
        "segment_curvature": [0.0] * n,
        "segment_angle": [90.0] * n,
        "segment_ncp": [1] * n,
        "segment_shape": [0.5] * n,
        "segment_square": [True] * n,
        "segment_square_shape": [1.0] * n,
        "segment_inflect": [False] * n,
        "segment_size": [0.5] * n,
        "segment_linetype": [1] * n,
        "segment_colour": [None] * n,
        "segment_alpha": [None] * n,
        "arrow_fill": [None] * n,
    })
    limits = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    return dict(
        data=df,
        lab=df["label"].tolist(),
        limits=limits,
        box_padding=Unit(0.25, "lines"),
        label_padding=Unit(0.25, "lines"),
        point_padding=Unit(1e-6, "lines"),
        label_r=Unit(0.15, "lines"),
        label_size=0.25,
        min_segment_length=Unit(0.5, "lines"),
        arrow=None,
        force=1.0,
        force_pull=1.0,
        max_time=0.1,
        max_iter=200,
        max_overlaps=20.0,
        direction="both",
        seed=seed,
        verbose=False,
    )


# --------------------------------------------------------------------------- factory

def test_factory_returns_layer():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "label": ["a", "b"]})
    layer = geom_label_repel(mapping=None, data=df, seed=7)
    assert type(layer).__name__ == "Layer"
    assert isinstance(layer.geom, GeomLabelRepel)


def test_factory_nudge_builds_position_nudge_repel():
    layer = geom_label_repel(nudge_x=0.4, nudge_y=-0.2)
    from ggrepel_py.position_nudge_repel import PositionNudgeRepel
    assert isinstance(layer.position, PositionNudgeRepel)
    assert layer.position.x == 0.4
    assert layer.position.y == -0.2


def test_factory_position_and_nudge_are_mutually_exclusive():
    with pytest.raises(ValueError, match="Use only one approach"):
        geom_label_repel(position=position_nudge_repel(0, 0), nudge_x=1)


# --------------------------------------------------------------------------- LabelRepelTree

def test_labelrepeltree_is_gtree_subclass():
    assert issubclass(LabelRepelTree, grid.GTree)


def test_labelrepeltree_make_content_populates_children():
    tree = LabelRepelTree(**_tree_inputs(n=5, seed=123))
    out = tree.make_content()
    assert out is tree
    assert tree.n_children() >= 1


def test_labelrepeltree_make_content_is_deterministic_under_seed():
    a = LabelRepelTree(**_tree_inputs(n=6, seed=77))
    b = LabelRepelTree(**_tree_inputs(n=6, seed=77))
    a.make_content()
    b.make_content()
    names_a = [g.name for g in a.get_children()]
    names_b = [g.name for g in b.get_children()]
    assert names_a == names_b


def test_labelrepeltree_segments_come_before_textboxes():
    tree = LabelRepelTree(**_tree_inputs(n=5, seed=1))
    tree.make_content()
    names = [g.name or "" for g in tree.get_children()]
    seg_positions = [i for i, n in enumerate(names) if n.startswith("segmentrepelgrob")]
    box_positions = [
        i for i, n in enumerate(names)
        if n.startswith("rectrepelgrob") or n.startswith("textrepelgrob")
    ]
    if seg_positions and box_positions:
        assert max(seg_positions) < min(box_positions)


def test_labelrepeltree_with_empty_labels_produces_no_grobs_for_blanks():
    inputs = _tree_inputs(n=4, seed=2)
    inputs["data"] = inputs["data"].copy()
    inputs["data"]["label"] = ["a", "", "c", ""]
    inputs["lab"] = ["a", "", "c", ""]
    tree = LabelRepelTree(**inputs)
    tree.make_content()
    names = [g.name or "" for g in tree.get_children()]
    rect_children = [n for n in names if n.startswith("rectrepelgrob")]
    text_children = [n for n in names if n.startswith("textrepelgrob")]
    # Only 2 non-empty labels → at most 2 rects + 2 texts.
    assert len(rect_children) <= 2
    assert len(text_children) <= 2


# --------------------------------------------------------------------------- defaults & aes

def test_geomlabelrepel_default_aes_has_snake_case_keys():
    aes = GeomLabelRepel.default_aes
    for key in ("point_size", "segment_colour", "segment_linetype",
                "segment_curvature", "linewidth", "fill"):
        assert key in aes


def test_geomlabelrepel_required_aes():
    assert set(GeomLabelRepel.required_aes) == {"x", "y", "label"}
