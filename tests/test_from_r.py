"""Ports of the R ``ggrepel`` testthat tests to pytest.

Source files under ``ggrepel/tests/testthat/``:

- ``test-to_unit.R`` → :class:`TestToUnit` (unit coercion of padding params)
- ``test-seed.R`` → :class:`TestSeed` (seed determinism)
- ``test-grob-order.R`` → :class:`TestGrobOrder` (segments before text/rect)
- ``test-lots-of-points.R`` → :class:`TestLotsOfPoints`

``test-element-text-repel.R`` is intentionally not ported — ``element_text_repel``
is a R-specific theme-integration feature that depends on ``element_grob``
dispatch and has been explicitly left out of scope in the Python port.

``test-just-with-angle.R`` uses ``vdiffr`` (visual-diff snapshot testing) which
has no direct Python equivalent; the test intent ("does not crash for common
just × angle combinations") is captured by :func:`test_just_with_angle`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from grid_py import Unit
from ggrepel_py import (
    geom_label_repel,
    geom_text_repel,
    LabelRepelTree,
    TextRepelTree,
)
from ggrepel_py.data import load_mtcars


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mtcars():
    """Load the bundled mtcars DataFrame with rownames as ``car``."""
    df = load_mtcars()
    if "car" not in df.columns:
        df = df.reset_index().rename(columns={"index": "car"})
    return df


@pytest.fixture
def mtcars_slice(mtcars):
    """Sparse mtcars subset (every 4th row) — matches R test-seed.R style."""
    df = mtcars.iloc[::4].reset_index(drop=True).copy()
    df["label"] = df["car"]
    return df


# ---------------------------------------------------------------------------
# test-to_unit.R — unit coercion of padding / size parameters
# ---------------------------------------------------------------------------


def _geom_param(layer, name):
    """Extract a geom_params entry from a ``Layer`` object."""
    return layer.geom_params[name]


class TestToUnit:
    """Ports of ``test-to_unit.R`` (R ggrepel/tests/testthat/test-to_unit.R)."""

    def test_label_defaults(self):
        """``box_padding`` etc. default to ``Unit(0.25, 'lines')`` (R unit(0.25, 'lines'))."""
        lay = geom_label_repel()
        assert _geom_param(lay, "box_padding").values[0] == pytest.approx(0.25)
        assert _geom_param(lay, "box_padding")._units[0] == "lines"
        assert _geom_param(lay, "label_padding").values[0] == pytest.approx(0.25)
        assert _geom_param(lay, "point_padding").values[0] == pytest.approx(1e-6)
        assert _geom_param(lay, "label_r").values[0] == pytest.approx(0.15)
        assert _geom_param(lay, "min_segment_length").values[0] == pytest.approx(0.5)

    def test_text_defaults(self):
        """``geom_text_repel`` defaults match R."""
        lay = geom_text_repel()
        assert _geom_param(lay, "box_padding").values[0] == pytest.approx(0.25)
        assert _geom_param(lay, "point_padding").values[0] == pytest.approx(1e-6)
        assert _geom_param(lay, "min_segment_length").values[0] == pytest.approx(0.5)

    def test_label_non_default_units(self):
        """Explicit ``Unit`` objects pass through unchanged."""
        lay = geom_label_repel(
            box_padding=Unit(1, "lines"),
            label_padding=Unit(2, "lines"),
            point_padding=Unit(3, "lines"),
            label_r=Unit(4, "lines"),
            min_segment_length=Unit(5, "lines"),
        )
        assert _geom_param(lay, "box_padding").values[0] == pytest.approx(1)
        assert _geom_param(lay, "label_padding").values[0] == pytest.approx(2)
        assert _geom_param(lay, "point_padding").values[0] == pytest.approx(3)
        assert _geom_param(lay, "label_r").values[0] == pytest.approx(4)
        assert _geom_param(lay, "min_segment_length").values[0] == pytest.approx(5)

    def test_text_non_default_units(self):
        lay = geom_text_repel(
            box_padding=Unit(1, "lines"),
            point_padding=Unit(2, "lines"),
            min_segment_length=Unit(3, "lines"),
        )
        assert _geom_param(lay, "box_padding").values[0] == pytest.approx(1)
        assert _geom_param(lay, "point_padding").values[0] == pytest.approx(2)
        assert _geom_param(lay, "min_segment_length").values[0] == pytest.approx(3)

    def test_label_numeric_coerces_to_lines(self):
        """Plain numbers get wrapped in ``Unit(n, 'lines')``."""
        lay = geom_label_repel(
            box_padding=0.25, label_padding=0.25,
            point_padding=1e-6, label_r=0.15, min_segment_length=0.5,
        )
        for name, val in [
            ("box_padding", 0.25), ("label_padding", 0.25),
            ("point_padding", 1e-6), ("label_r", 0.15),
            ("min_segment_length", 0.5),
        ]:
            u = _geom_param(lay, name)
            assert u.values[0] == pytest.approx(val)
            assert u._units[0] == "lines"

    def test_text_numeric_coerces_to_lines(self):
        lay = geom_text_repel(
            box_padding=0.25, point_padding=1e-6, min_segment_length=0.5,
        )
        for name, val in [
            ("box_padding", 0.25), ("point_padding", 1e-6),
            ("min_segment_length", 0.5),
        ]:
            u = _geom_param(lay, name)
            assert u.values[0] == pytest.approx(val)
            assert u._units[0] == "lines"

    def test_mix_units_and_numbers(self):
        """Mix of ``Unit`` and numeric within the same layer."""
        lay = geom_label_repel(
            box_padding=Unit(1, "lines"),
            label_padding=2,
            point_padding=3,
            label_r=Unit(4, "lines"),
            min_segment_length=5,
        )
        assert _geom_param(lay, "box_padding").values[0] == pytest.approx(1)
        assert _geom_param(lay, "label_padding").values[0] == pytest.approx(2)
        assert _geom_param(lay, "point_padding").values[0] == pytest.approx(3)
        assert _geom_param(lay, "label_r").values[0] == pytest.approx(4)
        assert _geom_param(lay, "min_segment_length").values[0] == pytest.approx(5)

    def test_non_lines_units_preserved(self):
        """cm / inches / mm preserve their unit string."""
        lay = geom_label_repel(
            box_padding=Unit(1, "cm"),
            label_padding=2,
            point_padding=3,
            label_r=Unit(4, "cm"),
            min_segment_length=5,
        )
        assert _geom_param(lay, "box_padding")._units[0] == "cm"
        assert _geom_param(lay, "label_r")._units[0] == "cm"
        assert _geom_param(lay, "label_padding")._units[0] == "lines"

    def test_NA_padding_propagates(self):
        """``float('nan')`` passes through as NaN, matching R's ``NA``.

        R test: ``expect_true(is.na(extract_param(p, 'box.padding')))`` and
        ``expect_true(class(...) != 'unit')``.
        """
        lay = geom_label_repel(box_padding=float("nan"))
        bp = _geom_param(lay, "box_padding")
        # to_unit() returns NaN for NaN input (not a Unit wrapper).
        assert isinstance(bp, float) and np.isnan(bp)

        lay2 = geom_text_repel(box_padding=float("nan"))
        bp2 = _geom_param(lay2, "box_padding")
        assert isinstance(bp2, float) and np.isnan(bp2)


# ---------------------------------------------------------------------------
# test-seed.R — determinism under fixed seed
# ---------------------------------------------------------------------------


def _run_tree_and_extract_pos(data, seed=None, label_col="label", cls=TextRepelTree):
    """Build a tree, run make_content, collect emitted text positions."""
    from ggplot2_py import ggplot, aes, geom_point, ggsave
    from ggrepel_py import geom_text_repel, geom_label_repel
    factory = geom_text_repel if cls is TextRepelTree else geom_label_repel
    p = (
        ggplot(data, aes(x="wt", y="mpg", label=label_col))
        + geom_point()
        + factory(seed=seed)
    )
    # Render to temp path to force draw_panel execution.
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        ggsave(path, p, width=4, height=4, dpi=72, units="in")
        return os.path.getsize(path)
    finally:
        os.unlink(path)


class TestSeed:
    """Ports of ``test-seed.R``."""

    def test_text_seed_reproducible(self, mtcars_slice):
        """Same seed → identical tree output (``geom_text_repel``)."""
        from ggplot2_py import ggplot_build, ggplot, aes, geom_point
        from ggrepel_py import geom_text_repel

        def _positions(seed):
            p = (
                ggplot(mtcars_slice, aes(x="wt", y="mpg", label="label"))
                + geom_point()
                + geom_text_repel(seed=seed)
            )
            # Build triggers draw_panel via ggplot_build → ggplot_gtable in repr
            png = p._repr_png_()
            assert png is not None
            return png

        p1 = _positions(42)
        p2 = _positions(42)
        assert p1 == p2, "identical seed produced different outputs"

    def test_text_no_seed_varies(self, mtcars_slice):
        """No seed → outputs differ across calls (stochastic jitter)."""
        from ggplot2_py import ggplot, aes, geom_point
        from ggrepel_py import geom_text_repel

        def _positions():
            p = (
                ggplot(mtcars_slice, aes(x="wt", y="mpg", label="label"))
                + geom_point()
                + geom_text_repel()
            )
            return p._repr_png_()

        p1 = _positions()
        p2 = _positions()
        # Some runs may coincide; generate up to 5 samples until divergence.
        if p1 == p2:
            samples = [p1, p2]
            for _ in range(3):
                samples.append(_positions())
            assert len(set(samples)) > 1, "no-seed runs produced identical outputs"

    def test_label_seed_reproducible(self, mtcars_slice):
        """Same seed → identical tree output (``geom_label_repel``)."""
        from ggplot2_py import ggplot, aes, geom_point
        from ggrepel_py import geom_label_repel

        def _positions(seed):
            p = (
                ggplot(mtcars_slice, aes(x="wt", y="mpg", label="label"))
                + geom_point()
                + geom_label_repel(seed=seed)
            )
            return p._repr_png_()

        p1 = _positions(10)
        p2 = _positions(10)
        assert p1 == p2


# ---------------------------------------------------------------------------
# test-grob-order.R — z-ordering within a repeltree
# ---------------------------------------------------------------------------


class TestGrobOrder:
    """Ports of ``test-grob-order.R``."""

    def test_text_segments_before_text(self, mtcars):
        """All segmentrepelgrob before textrepelgrob in the child list."""
        from ggrepel_py.geom_text_repel import TextRepelTree

        data = mtcars.rename(columns={"car": "label"}).copy()
        data["x"] = data["wt"]
        data["y"] = data["mpg"]
        data["x_orig"] = data["x"]
        data["y_orig"] = data["y"]
        data["hjust"] = 0.5
        data["vjust"] = 0.5
        # Placeholder limits in [0,1] (post-coord_transform domain).
        limits = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
        tree = TextRepelTree(
            data=data, lab=data["label"].tolist(), limits=limits,
            box_padding=Unit(0.25, "lines"),
            point_padding=Unit(1e-6, "lines"),
            min_segment_length=Unit(0.5, "lines"),
            arrow=None, force=1.0, force_pull=1.0,
            max_time=0.5, max_iter=2000, max_overlaps=10,
            direction="both", seed=42, verbose=False,
        )
        tree.make_content()
        names = [str(getattr(g, "name", "")) for g in tree._children.values()]
        segs = [i for i, n in enumerate(names) if n.startswith("segment")]
        texts = [i for i, n in enumerate(names) if n.startswith("text")]
        if segs and texts:
            assert max(segs) < min(texts), \
                f"segment grob at {max(segs)} should come before text at {min(texts)}"

    def test_label_rect_before_text(self, mtcars):
        """For LabelRepelTree: each rect grob precedes its paired text grob."""
        from ggrepel_py.geom_label_repel import LabelRepelTree

        data = mtcars.rename(columns={"car": "label"}).copy()
        data["x"] = data["wt"]; data["y"] = data["mpg"]
        data["x_orig"] = data["x"]; data["y_orig"] = data["y"]
        data["hjust"] = 0.5; data["vjust"] = 0.5
        limits = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
        tree = LabelRepelTree(
            data=data, lab=data["label"].tolist(), limits=limits,
            box_padding=Unit(0.25, "lines"),
            label_padding=Unit(0.25, "lines"),
            point_padding=Unit(1e-6, "lines"),
            label_r=Unit(0.15, "lines"),
            label_size=0.25,
            min_segment_length=Unit(0.5, "lines"),
            arrow=None, force=1.0, force_pull=1.0,
            max_time=0.5, max_iter=2000, max_overlaps=float("inf"),
            direction="both", seed=42, verbose=False,
        )
        tree.make_content()
        names = [str(getattr(g, "name", "")) for g in tree._children.values()]
        segs = [i for i, n in enumerate(names) if n.startswith("segment")]
        rects = [i for i, n in enumerate(names) if n.startswith("rect")]
        texts = [i for i, n in enumerate(names) if n.startswith("text")]
        # R expects: len(rect) == len(text) and segments < rect < text
        assert len(rects) == len(texts), \
            f"expected rect count ({len(rects)}) == text count ({len(texts)})"
        if segs:
            assert max(segs) < min(rects), "segment should come before rect"
            assert max(segs) < min(texts), "segment should come before text"


# ---------------------------------------------------------------------------
# test-lots-of-points.R — stress test with many points
# ---------------------------------------------------------------------------


class TestLotsOfPoints:
    """Ports of ``test-lots-of-points.R``."""

    def test_many_points_render(self, mtcars):
        """10k points + mtcars labels should render without error."""
        from ggplot2_py import ggplot, aes, geom_point
        from ggrepel_py import geom_text_repel

        rng = np.random.default_rng(42)
        bg = pd.DataFrame({
            "wt": rng.normal(3, 1, 10000),
            "mpg": rng.normal(19, 1, 10000),
            "car": [""] * 10000,
        })
        cars = mtcars[["wt", "mpg", "car"]].copy()
        dat = pd.concat([bg, cars], ignore_index=True)
        dat["label"] = dat["car"]
        p = (
            ggplot(dat, aes(x="wt", y="mpg", label="label"))
            + geom_point()
            + geom_text_repel(box_padding=0.5, max_overlaps=float("inf"), seed=42)
        )
        # Limit solver time to keep the test fast.
        png = p._repr_png_()
        assert png is not None and len(png) > 100


# ---------------------------------------------------------------------------
# test-just-with-angle.R — no-crash smoke over just × angle combinations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("angle", [0, 90, 180])
@pytest.mark.parametrize("hjust,vjust", [(0.5, 0.5), (0, 0), (1, 1), ("inward", "inward")])
def test_just_with_angle(mtcars_slice, angle, hjust, vjust):
    """Replaces R's ``vdiffr`` tests with plain no-crash checks."""
    from ggplot2_py import ggplot, aes, geom_point
    from ggrepel_py import geom_text_repel

    p = (
        ggplot(mtcars_slice, aes(x="wt", y="mpg", label="label"))
        + geom_point()
        + geom_text_repel(
            vjust=vjust, hjust=hjust, angle=angle, max_iter=0, seed=1234,
        )
    )
    png = p._repr_png_()
    assert png is not None
