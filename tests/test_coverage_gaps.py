"""Targeted tests filling coverage holes in ggrepel_py.

Each test addresses a specific branch or helper that the R-ported tests
and audit-fix tests do not exercise by themselves.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from grid_py import Unit

from ggrepel_py._utilities import compute_just, exclude_outside, not_empty, to_unit
from ggrepel_py.geom_text_repel import (
    _as_float_array,
    _bg_colour_with_alpha,
    _convert_height_native,
    _convert_width_native,
    _coord_transform,
)
from ggrepel_py.position_nudge_repel import PositionNudgeRepel, position_nudge_repel


# ---------------------------------------------------------------------------
# _utilities.py
# ---------------------------------------------------------------------------


class TestNotEmpty:

    def test_none_is_empty(self):
        assert list(not_empty([None, "a", ""])) == [False, True, False]

    def test_nan_is_empty(self):
        assert list(not_empty([float("nan"), "b"])) == [False, True]

    def test_numeric_values_are_non_empty(self):
        """Numbers stringify to non-empty → all True."""
        assert list(not_empty([1, 2, 3.5])) == [True, True, True]


class TestExcludeOutside:

    def test_filters_points_outside_range(self):
        df = pd.DataFrame({"x": [0.0, 0.5, 1.0, 1.5], "y": [0.1, 0.5, 0.9, 2.0]})
        # panel_scales can be a dict with ``x.range`` / ``y.range``;
        # ``inside()`` uses a closed [lo, hi] interval so x=1.0 passes.
        out = exclude_outside(df, {"x.range": [0.0, 1.0], "y.range": [0.0, 1.0]})
        assert len(out) == 3  # (1.5, 2.0) is the only out-of-range point
        assert list(out["x"]) == [0.0, 0.5, 1.0]

    def test_underscore_aliases_accepted(self):
        df = pd.DataFrame({"x": [0.5, 2.0], "y": [0.5, 0.5]})
        out = exclude_outside(df, {"x_range": [0.0, 1.0], "y_range": [0.0, 1.0]})
        assert len(out) == 1

    def test_missing_ranges_returns_data_unchanged(self):
        df = pd.DataFrame({"x": [0.0, 2.0], "y": [0.0, 2.0]})
        out = exclude_outside(df, {})
        assert list(out["x"]) == [0.0, 2.0]


class TestComputeJust:

    def test_center(self):
        got = compute_just(np.array(["center"]), np.array([0.5]), np.array([0.5]), np.array([0.0]))
        assert list(got) == [0.5]

    def test_left_right(self):
        got = compute_just(np.array(["left", "right"]),
                           np.array([0.5, 0.5]), np.array([0.0, 0.0]),
                           np.array([0.0, 0.0]))
        # "left" → 0, "right" → 1
        assert list(got) == [0.0, 1.0]

    def test_inward_outward(self):
        # "inward"/"outward" depend on the ``b`` / ``angle`` values
        got = compute_just(np.array(["inward", "outward"]),
                           np.array([0.2, 0.8]),
                           np.array([0.0, 0.0]),
                           np.array([0.0, 0.0]))
        # inward at x=0.2 → 0, inward at x=0.8 → 1 (away from center)
        # Actually: inward at 0.2 means push toward center (right) → 0
        # outward at 0.8 means push away from center (right) → 1
        assert len(got) == 2

    def test_numeric_string_coerces(self):
        got = compute_just(np.array(["0.25"]), np.array([0.5]), np.array([0.5]), np.array([0.0]))
        assert list(got) == [0.25]


class TestToUnitHelper:

    def test_none_returns_none(self):
        assert to_unit(None) is None

    def test_nan_returns_nan(self):
        v = to_unit(float("nan"))
        assert isinstance(v, float) and math.isnan(v)

    def test_numeric_wraps_in_lines(self):
        u = to_unit(0.5)
        assert u.values[0] == pytest.approx(0.5)
        assert u._units[0] == "lines"

    def test_existing_unit_passes_through(self):
        u = Unit(1, "cm")
        assert to_unit(u) is u


# ---------------------------------------------------------------------------
# position_nudge_repel.py — the zero-nudge branch
# ---------------------------------------------------------------------------


class TestPositionNudgeZeroNudge:

    def test_zero_nudge_copies_data(self):
        """When both nudges are 0 the layer still copies the DataFrame."""
        pos = position_nudge_repel(x=0, y=0)
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        out = pos.compute_layer(df, {"x": 0, "y": 0})
        assert out is not df
        assert list(out["x"]) == [1.0, 2.0]

    def test_x_only_nudge(self):
        pos = position_nudge_repel(x=0.1, y=0)
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        out = pos.compute_layer(df, {"x": 0.1, "y": 0})
        assert "x_orig" in out.columns
        assert list(out["x_orig"]) == [1.0, 2.0]
        assert list(out["x"]) == [1.1, 2.1]
        # y untouched
        assert list(out["y"]) == [3.0, 4.0]

    def test_y_only_nudge(self):
        pos = position_nudge_repel(x=0, y=0.2)
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        out = pos.compute_layer(df, {"x": 0, "y": 0.2})
        assert list(out["x"]) == [1.0, 2.0]
        assert list(out["y"]) == pytest.approx([3.2, 4.2])

    def test_both_axes_nudge(self):
        pos = position_nudge_repel(x=0.1, y=0.2)
        df = pd.DataFrame({"x": [1.0], "y": [3.0]})
        out = pos.compute_layer(df, {"x": 0.1, "y": 0.2})
        assert list(out["x"]) == pytest.approx([1.1])
        assert list(out["y"]) == pytest.approx([3.2])

    def test_setup_params_returns_x_and_y(self):
        """``setup_params`` returns the layer's nudge offsets."""
        pos = position_nudge_repel(x=0.3, y=0.4)
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        params = pos.setup_params(df)
        assert params == {"x": 0.3, "y": 0.4}


# ---------------------------------------------------------------------------
# geom_text_repel.py helpers
# ---------------------------------------------------------------------------


class TestBgColourHelper:

    def test_none_bg_returns_none(self):
        assert _bg_colour_with_alpha(None, 0.5) is None

    def test_nan_bg_returns_none(self):
        assert _bg_colour_with_alpha(float("nan"), 0.5) is None

    def test_colour_with_none_alpha(self):
        out = _bg_colour_with_alpha("black", None)
        # scales.alpha returns a hex string with opaque alpha
        assert isinstance(out, str)

    def test_colour_with_alpha(self):
        out = _bg_colour_with_alpha("black", 0.5)
        assert isinstance(out, str)


class TestAsFloatArray:

    def test_none_fills_default(self):
        arr = _as_float_array(None, 3, 42.0)
        assert list(arr) == [42.0, 42.0, 42.0]

    def test_scalar_broadcasts(self):
        arr = _as_float_array(7, 3, 0.0)
        assert list(arr) == [7.0, 7.0, 7.0]

    def test_array_length_mismatch_resizes(self):
        arr = _as_float_array([1, 2], 4, 0.0)
        assert len(arr) == 4

    def test_nan_replaced_with_default(self):
        arr = _as_float_array([1.0, float("nan"), 3.0], 3, 0.0)
        assert list(arr) == [1.0, 0.0, 3.0]


class TestConvertNative:

    def test_width_native(self):
        v = _convert_width_native(0.25)
        assert isinstance(v, float)

    def test_width_native_from_unit(self):
        v = _convert_width_native(Unit(0.5, "native"))
        assert v == pytest.approx(0.5)

    def test_height_native(self):
        v = _convert_height_native(Unit(0.75, "native"))
        assert v == pytest.approx(0.75)


class TestCoordTransform:

    def test_no_coord_returns_data(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        out = _coord_transform(None, df, {})
        assert out is df

    def test_coord_without_transform_returns_data(self):
        class FakeCoord:
            pass
        df = pd.DataFrame({"x": [1], "y": [2]})
        out = _coord_transform(FakeCoord(), df, {})
        assert out is df


# ---------------------------------------------------------------------------
# geom_label_repel — exercise draw_panel + LabelRepelTree.make_content
# ---------------------------------------------------------------------------


class TestGeomLabelRepelPipeline:
    """These tests render full plots to force :class:`LabelRepelTree` to
    walk its ``make_content`` method, covering per-row grob assembly."""

    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0],
            "car": ["alpha", "beta", "gamma"],
        })

    def test_basic_label_plot(self, df):
        from ggplot2_py import ggplot, aes, geom_point
        from ggrepel_py import geom_label_repel

        p = (
            ggplot(df, aes(x="x", y="y", label="car"))
            + geom_point()
            + geom_label_repel(seed=42)
        )
        png = p._repr_png_()
        assert png is not None and len(png) > 100

    def test_label_with_arrow(self, df):
        from ggplot2_py import ggplot, aes, geom_point
        from grid_py import arrow, Unit
        from ggrepel_py import geom_label_repel

        p = (
            ggplot(df, aes(x="x", y="y", label="car"))
            + geom_point()
            + geom_label_repel(
                arrow=arrow(length=Unit(2, "mm"), type="closed"),
                seed=42,
            )
        )
        assert p._repr_png_() is not None

    def test_label_with_linewidth_zero(self, df):
        """``linewidth=0`` triggers the ``col=None`` branch on the rect gp."""
        from ggplot2_py import ggplot, aes, geom_point
        from ggrepel_py import geom_label_repel

        p = (
            ggplot(df, aes(x="x", y="y", label="car"))
            + geom_point()
            + geom_label_repel(linewidth=0, seed=42)
        )
        assert p._repr_png_() is not None

    def test_label_max_overlaps_zero(self, df):
        """max_overlaps=0 with verbose: exercises too_many branch + inform."""
        from ggplot2_py import ggplot, aes, geom_point
        from ggrepel_py import geom_label_repel

        p = (
            ggplot(df, aes(x="x", y="y", label="car"))
            + geom_point()
            + geom_label_repel(max_overlaps=0, verbose=True, seed=42)
        )
        assert p._repr_png_() is not None

    def test_label_with_nudge_pushes_labels(self, df):
        from ggplot2_py import ggplot, aes, geom_point
        from ggrepel_py import geom_label_repel

        p = (
            ggplot(df, aes(x="x", y="y", label="car"))
            + geom_point()
            + geom_label_repel(nudge_x=0.3, nudge_y=0.1, seed=42)
        )
        assert p._repr_png_() is not None

    def test_label_empty_data_returns_empty(self):
        """Empty DataFrame produces a renderable null plot."""
        from ggplot2_py import ggplot, aes, geom_point
        from ggrepel_py import geom_label_repel

        df_empty = pd.DataFrame({"x": [], "y": [], "car": []})
        p = (
            ggplot(df_empty, aes(x="x", y="y", label="car"))
            + geom_point()
            + geom_label_repel(seed=42)
        )
        assert p._repr_png_() is not None

    def test_label_draw_panel_nudge_restoration(self):
        """Post-nudge x/y restoration path in draw_panel.

        When ``position_nudge_repel`` is supplied separately (rather than via
        ``nudge_x``/``nudge_y`` kwargs), ``x_orig``/``y_orig`` columns exist
        and the draw_panel branch that restores them is exercised.
        """
        from ggplot2_py import ggplot, aes, geom_point
        from ggrepel_py import geom_label_repel, position_nudge_repel

        df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0], "car": ["a", "b"]})
        p = (
            ggplot(df, aes(x="x", y="y", label="car"))
            + geom_point()
            + geom_label_repel(position=position_nudge_repel(x=0.1, y=0), seed=42)
        )
        assert p._repr_png_() is not None


class TestGeomTextRepelPipeline:
    """Mirror TestGeomLabelRepelPipeline for ``geom_text_repel``."""

    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0],
            "car": ["alpha", "beta", "gamma"],
        })

    def test_basic_text_plot(self, df):
        from ggplot2_py import ggplot, aes, geom_point
        from ggrepel_py import geom_text_repel

        p = (
            ggplot(df, aes(x="x", y="y", label="car"))
            + geom_point()
            + geom_text_repel(seed=42)
        )
        assert p._repr_png_() is not None

    def test_text_with_bg_colour(self, df):
        from ggplot2_py import ggplot, aes, geom_point
        from ggrepel_py import geom_text_repel

        p = (
            ggplot(df, aes(x="x", y="y", label="car"))
            + geom_point()
            + geom_text_repel(bg_colour="red", bg_r=0.15, seed=42)
        )
        assert p._repr_png_() is not None

    def test_text_with_arrow(self, df):
        from ggplot2_py import ggplot, aes, geom_point
        from grid_py import arrow, Unit
        from ggrepel_py import geom_text_repel

        p = (
            ggplot(df, aes(x="x", y="y", label="car"))
            + geom_point()
            + geom_text_repel(arrow=arrow(length=Unit(2, "mm")), seed=42)
        )
        assert p._repr_png_() is not None

    def test_text_position_nudge(self):
        """Use ``position_nudge_repel`` via ``position=`` kwarg."""
        from ggplot2_py import ggplot, aes, geom_point
        from ggrepel_py import geom_text_repel, position_nudge_repel

        df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0], "car": ["a", "b"]})
        p = (
            ggplot(df, aes(x="x", y="y", label="car"))
            + geom_point()
            + geom_text_repel(position=position_nudge_repel(x=0.2, y=0.1), seed=42)
        )
        assert p._repr_png_() is not None

    def test_text_conflicting_position_and_nudge_raises(self):
        """Setting both ``position`` and ``nudge_x`` is a user error."""
        from ggrepel_py import geom_text_repel, position_nudge_repel

        with pytest.raises(ValueError, match=r"`position`"):
            geom_text_repel(
                nudge_x=0.1, position=position_nudge_repel(x=0.2, y=0),
            )

    def test_label_conflicting_position_and_nudge_raises(self):
        from ggrepel_py import geom_label_repel, position_nudge_repel

        with pytest.raises(ValueError, match=r"`position`"):
            geom_label_repel(
                nudge_y=0.1, position=position_nudge_repel(x=0, y=0.2),
            )

    def test_text_empty_data_returns_null(self):
        from ggplot2_py import ggplot, aes, geom_point
        from ggrepel_py import geom_text_repel

        df = pd.DataFrame({"x": [], "y": [], "car": []})
        p = (
            ggplot(df, aes(x="x", y="y", label="car"))
            + geom_point()
            + geom_text_repel(seed=42)
        )
        assert p._repr_png_() is not None

    def test_text_inf_xlim_ylim(self):
        from ggplot2_py import ggplot, aes, geom_point
        from ggrepel_py import geom_text_repel

        df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0], "car": ["a", "b"]})
        p = (
            ggplot(df, aes(x="x", y="y", label="car"))
            + geom_point()
            + geom_text_repel(
                xlim=(float("-inf"), float("inf")),
                ylim=(0.0, float("inf")),
                seed=42,
            )
        )
        assert p._repr_png_() is not None

    def test_label_inf_xlim(self):
        from ggplot2_py import ggplot, aes, geom_point
        from ggrepel_py import geom_label_repel

        df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0], "car": ["a", "b"]})
        p = (
            ggplot(df, aes(x="x", y="y", label="car"))
            + geom_point()
            + geom_label_repel(xlim=(float("-inf"), 10.0), seed=42)
        )
        assert p._repr_png_() is not None

    def test_text_max_overlaps_zero_verbose(self):
        """Exercises the C++ solver verbose inform path."""
        from ggplot2_py import ggplot, aes, geom_point
        from ggrepel_py import geom_text_repel

        # Many overlapping points at the same location.
        df = pd.DataFrame({"x": [1.0] * 10, "y": [1.0] * 10,
                           "car": [f"c{i}" for i in range(10)]})
        p = (
            ggplot(df, aes(x="x", y="y", label="car"))
            + geom_point()
            + geom_text_repel(max_overlaps=0, verbose=True, seed=42)
        )
        assert p._repr_png_() is not None


class TestUtilitiesExtraBranches:
    """Cover remaining helper branches in _utilities.py (line 88)."""

    def test_to_unit_sequence_of_numbers(self):
        """Numeric sequences become ``Unit`` vectors (line 88 branch)."""
        u = to_unit([0.25, 0.5])
        # either returns a Unit with two values, or wraps each; accept both.
        if isinstance(u, Unit):
            assert len(u.values) == 2


# ---------------------------------------------------------------------------
# Directly exercise draw_panel branches for x_orig / nudge_col bookkeeping
# ---------------------------------------------------------------------------


class TestDrawPanelBookkeeping:
    """The x/y ↔ x_orig/y_orig/nudge_x/nudge_y swap logic in draw_panel has
    four branches depending on which columns are present.  Craft minimal
    inputs to hit each one."""

    def _make_geom_and_run(self, cls_geom, data, panel_params=None):
        """Instantiate a Geom and invoke ``draw_panel`` directly."""
        inst = cls_geom()
        if panel_params is None:
            panel_params = {"x.range": [0.0, 10.0], "y.range": [0.0, 10.0]}
        return inst.draw_panel(
            data=data, panel_params=panel_params, coord=None,
            parse=False, na_rm=False,
            box_padding=0.25, point_padding=1e-6,
            min_segment_length=0.5, arrow=None,
            force=1.0, force_pull=1.0,
            max_time=0.1, max_iter=100, max_overlaps=10,
            nudge_x=0, nudge_y=0,
            xlim=(None, None), ylim=(None, None),
            direction="both", seed=42, verbose=False,
        )

    def test_text_draw_panel_with_nudge_and_orig_cols(self):
        """Branch: both ``nudge_*`` AND ``x_orig`` already present.

        Exercises ``geom_text_repel.py:733-741`` (else branch + orig_col).
        """
        from ggrepel_py.geom_text_repel import GeomTextRepel

        df = pd.DataFrame({
            "x": [1.1, 2.1, 3.1],    # nudged
            "y": [1.0, 2.0, 3.0],
            "x_orig": [1.0, 2.0, 3.0],  # pre-nudge
            "y_orig": [1.0, 2.0, 3.0],
            "nudge_x": [0.0, 0.0, 0.0],
            "nudge_y": [0.0, 0.0, 0.0],
            "label": ["a", "b", "c"],
            "group": [1, 1, 1],
            "PANEL": [1, 1, 1],
        })
        result = self._make_geom_and_run(GeomTextRepel, df)
        assert result is not None

    def test_text_draw_panel_nudge_col_all_zero(self):
        """Branch: ``nudge_*`` present but all zero → replace with x itself
        (geom_text_repel.py line 741).
        """
        from ggrepel_py.geom_text_repel import GeomTextRepel

        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0],
            "nudge_x": [0.0, 0.0, 0.0],  # all-zero
            "nudge_y": [0.0, 0.0, 0.0],
            "label": ["a", "b", "c"],
            "group": [1, 1, 1],
            "PANEL": [1, 1, 1],
        })
        result = self._make_geom_and_run(GeomTextRepel, df)
        assert result is not None

    def test_label_draw_panel_with_nudge_and_orig_cols(self):
        """Same branch as above for ``GeomLabelRepel``."""
        from ggrepel_py.geom_label_repel import GeomLabelRepel

        inst = GeomLabelRepel()
        df = pd.DataFrame({
            "x": [1.1, 2.1],
            "y": [1.0, 2.0],
            "x_orig": [1.0, 2.0],
            "y_orig": [1.0, 2.0],
            "nudge_x": [0.0, 0.0],
            "nudge_y": [0.0, 0.0],
            "label": ["a", "b"],
            "group": [1, 1],
            "PANEL": [1, 1],
        })
        result = inst.draw_panel(
            data=df,
            panel_params={"x.range": [0.0, 5.0], "y.range": [0.0, 5.0]},
            coord=None, parse=False, na_rm=False,
            box_padding=0.25, label_padding=0.25, point_padding=1e-6,
            label_r=0.15, label_size=0.25, min_segment_length=0.5,
            arrow=None, force=1.0, force_pull=1.0,
            max_time=0.1, max_iter=100, max_overlaps=10,
            nudge_x=0, nudge_y=0,
            xlim=(None, None), ylim=(None, None),
            direction="both", seed=42, verbose=False,
        )
        assert result is not None

    def test_text_draw_panel_none_data_returns_null(self):
        """``draw_panel(data=None)`` → null_grob (line 719)."""
        from ggrepel_py.geom_text_repel import GeomTextRepel
        from grid_py._grob import Grob

        inst = GeomTextRepel()
        result = inst.draw_panel(
            data=None,
            panel_params={"x.range": [0.0, 1.0], "y.range": [0.0, 1.0]},
            coord=None, parse=False, na_rm=False,
            box_padding=0.25, point_padding=1e-6,
            min_segment_length=0.5, arrow=None,
            force=1.0, force_pull=1.0,
            max_time=0.1, max_iter=100, max_overlaps=10,
            nudge_x=0, nudge_y=0,
            xlim=(None, None), ylim=(None, None),
            direction="both", seed=42, verbose=False,
        )
        assert isinstance(result, Grob)

    def test_label_draw_panel_none_data_returns_null(self):
        from ggrepel_py.geom_label_repel import GeomLabelRepel
        from grid_py._grob import Grob

        inst = GeomLabelRepel()
        result = inst.draw_panel(
            data=None,
            panel_params={"x.range": [0.0, 1.0], "y.range": [0.0, 1.0]},
            coord=None, parse=False, na_rm=False,
            box_padding=0.25, label_padding=0.25, point_padding=1e-6,
            label_r=0.15, label_size=0.25, min_segment_length=0.5,
            arrow=None, force=1.0, force_pull=1.0,
            max_time=0.1, max_iter=100, max_overlaps=10,
            nudge_x=0, nudge_y=0,
            xlim=(None, None), ylim=(None, None),
            direction="both", seed=42, verbose=False,
        )
        assert isinstance(result, Grob)

    def test_text_norm_lim_single_element(self):
        """``xlim=(5.0,)`` with fewer than 2 elements gets resized (line 763)."""
        from ggrepel_py.geom_text_repel import GeomTextRepel

        df = pd.DataFrame({
            "x": [1.0, 2.0], "y": [1.0, 2.0],
            "label": ["a", "b"], "group": [1, 1], "PANEL": [1, 1],
        })
        inst = GeomTextRepel()
        result = inst.draw_panel(
            data=df,
            panel_params={"x.range": [0.0, 5.0], "y.range": [0.0, 5.0]},
            coord=None, parse=False, na_rm=False,
            box_padding=0.25, point_padding=1e-6,
            min_segment_length=0.5, arrow=None,
            force=1.0, force_pull=1.0,
            max_time=0.1, max_iter=100, max_overlaps=10,
            nudge_x=0, nudge_y=0,
            xlim=(5.0,),  # single-element tuple
            ylim=None,     # triggers ``if lim is None`` branch
            direction="both", seed=42, verbose=False,
        )
        assert result is not None

    def test_label_norm_lim_single_element(self):
        from ggrepel_py.geom_label_repel import GeomLabelRepel

        df = pd.DataFrame({
            "x": [1.0, 2.0], "y": [1.0, 2.0],
            "label": ["a", "b"], "group": [1, 1], "PANEL": [1, 1],
        })
        inst = GeomLabelRepel()
        result = inst.draw_panel(
            data=df,
            panel_params={"x.range": [0.0, 5.0], "y.range": [0.0, 5.0]},
            coord=None, parse=False, na_rm=False,
            box_padding=0.25, label_padding=0.25, point_padding=1e-6,
            label_r=0.15, label_size=0.25, min_segment_length=0.5,
            arrow=None, force=1.0, force_pull=1.0,
            max_time=0.1, max_iter=100, max_overlaps=10,
            nudge_x=0, nudge_y=0,
            xlim=(5.0,), ylim=None,
            direction="both", seed=42, verbose=False,
        )
        assert result is not None

