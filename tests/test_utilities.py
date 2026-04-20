"""Ports of ggrepel's test-to_unit.R plus coverage for compute_just/just_dir."""

from __future__ import annotations

import math

import numpy as np
import pytest

import grid_py as grid
from ggrepel_py._utilities import (
    PT,
    compute_just,
    inside,
    just_dir,
    null_default,
    to_unit,
)


# ---- PT and null_default ----------------------------------------------------

def test_pt_value():
    assert PT == pytest.approx(72.27 / 25.4)


def test_null_default():
    assert null_default("x", "y") == "x"
    assert null_default(None, "y") == "y"
    assert null_default(0, "fallback") == 0


# ---- to_unit (ports of test-to_unit.R) --------------------------------------

def test_to_unit_numeric_becomes_lines_unit():
    out = to_unit(1)
    assert grid.is_unit(out)


def test_to_unit_preserves_existing_unit():
    u = grid.Unit(0.5, "npc")
    assert to_unit(u) is u


def test_to_unit_returns_nan_for_scalar_nan():
    out = to_unit(float("nan"))
    assert isinstance(out, float) and math.isnan(out)


def test_to_unit_none():
    assert to_unit(None) is None


# ---- inside -----------------------------------------------------------------

def test_inside_basic():
    arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    mask = inside(arr, (1.0, 3.0))
    np.testing.assert_array_equal(mask, [False, True, True, True, False])


def test_inside_passes_through_infinities():
    arr = np.array([-np.inf, 0.5, np.inf])
    mask = inside(arr, (0.0, 1.0))
    np.testing.assert_array_equal(mask, [True, True, True])


# ---- just_dir ---------------------------------------------------------------

def test_just_dir_bands():
    arr = np.array([0.0, 0.5, 1.0])
    np.testing.assert_array_equal(just_dir(arr), [1, 2, 3])


# ---- compute_just (a subset of test-just-with-angle.R) ----------------------

def test_compute_just_named_strings():
    out = compute_just(["left", "center", "right"], [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(out, [0.0, 0.5, 1.0])


def test_compute_just_bottom_middle_top():
    out = compute_just(["bottom", "middle", "top"], [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(out, [0.0, 0.5, 1.0])


def test_compute_just_inward_no_rotation():
    # 0-degree angle: "inward" → left-or-right depending on a vs 0.5.
    out = compute_just(["inward", "inward", "inward"], [0.2, 0.5, 0.8], angle=0)
    np.testing.assert_array_equal(out, [0.0, 0.5, 1.0])


def test_compute_just_outward_no_rotation():
    out = compute_just(["outward", "outward", "outward"], [0.2, 0.5, 0.8], angle=0)
    np.testing.assert_array_equal(out, [1.0, 0.5, 0.0])


def test_compute_just_inward_rotated_90_deg():
    # 90-degree rotation swaps the axis: inward on the "b" coordinate.
    out = compute_just(
        ["inward", "inward", "inward"],
        [0.5, 0.5, 0.5],
        [0.2, 0.5, 0.8],
        angle=90,
    )
    np.testing.assert_array_equal(out, [0.0, 0.5, 1.0])
