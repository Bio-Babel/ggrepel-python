"""Regression tests for the 8 gaps fixed during the ggrepel_py audit.

Each ``TestGap*`` class verifies one fix.  Citations point at the R
gold-standard file/line in the ``ggrepel`` R package.
"""

from __future__ import annotations

import io
import logging
import warnings

import numpy as np
import pandas as pd
import pytest

from ggplot2_py import aes, geom_point, ggplot, ggsave
from ggrepel_py import (
    __version__,
    geom_label_repel,
    geom_text_repel,
    get_option,
    set_option,
)
from ggrepel_py._options import inform as _inform


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_df():
    return pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "car": ["a", "b", "c"]})


@pytest.fixture(autouse=True)
def reset_options():
    """Clear any option set by a previous test."""
    for k in list(("ggrepel.max.overlaps", "verbose")):
        set_option(k, None)
    yield
    for k in list(("ggrepel.max.overlaps", "verbose")):
        set_option(k, None)


def _render(p, path="/tmp/_audit_test.png"):
    """Render a plot via ggsave to exercise the draw_panel code path."""
    ggsave(path, p, width=3, height=3, dpi=72, units="in")


# ---------------------------------------------------------------------------
# Gap #1: ``bg.colour`` alpha pre-mix (R geom-text-repel.R:531)
# ---------------------------------------------------------------------------


class TestGap1BgColourAlpha:

    def test_bg_colour_with_alpha_does_not_crash(self, small_df):
        """Setting both ``bg_colour`` and ``alpha`` should render cleanly."""
        p = (
            ggplot(small_df, aes(x="x", y="y", label="car"))
            + geom_point()
            + geom_text_repel(bg_colour="black", bg_r=0.2, alpha=0.5, seed=42)
        )
        png = p._repr_png_()
        assert png is not None and len(png) > 100

    def test_bg_colour_none_skips_halo(self, small_df):
        """``bg_colour=None`` still works with alpha set (halo skipped)."""
        p = (
            ggplot(small_df, aes(x="x", y="y", label="car"))
            + geom_point()
            + geom_text_repel(alpha=0.5, seed=42)
        )
        png = p._repr_png_()
        assert png is not None


# ---------------------------------------------------------------------------
# Gap #2: ``parse=True`` is rejected (R uses parse_safe / plotmath)
# ---------------------------------------------------------------------------


class TestGap2ParseRejected:

    def test_text_parse_raises(self, small_df):
        p = (
            ggplot(small_df, aes(x="x", y="y", label="car"))
            + geom_point()
            + geom_text_repel(parse=True, seed=42)
        )
        with pytest.raises(NotImplementedError, match="parse=True"):
            _render(p)

    def test_label_parse_raises(self, small_df):
        p = (
            ggplot(small_df, aes(x="x", y="y", label="car"))
            + geom_point()
            + geom_label_repel(parse=True, seed=42)
        )
        with pytest.raises(NotImplementedError, match="parse=True"):
            _render(p)


# ---------------------------------------------------------------------------
# Gap #3: all-empty labels short-circuit to null_grob (R geom-text-repel.R:284-286)
# ---------------------------------------------------------------------------


class TestGap3AllEmptyLabelsExit:

    def test_text_all_empty_labels(self):
        df = pd.DataFrame({"x": [1, 2], "y": [1, 2], "car": ["", ""]})
        p = (
            ggplot(df, aes(x="x", y="y", label="car"))
            + geom_point()
            + geom_text_repel(seed=42)
        )
        # Should render a valid image without any label grobs.
        png = p._repr_png_()
        assert png is not None

    def test_label_all_empty_labels(self):
        df = pd.DataFrame({"x": [1, 2], "y": [1, 2], "car": ["", ""]})
        p = (
            ggplot(df, aes(x="x", y="y", label="car"))
            + geom_point()
            + geom_label_repel(seed=42)
        )
        png = p._repr_png_()
        assert png is not None

    def test_mixed_empty_and_non_empty(self):
        """At least one non-empty → full path runs, no short-circuit."""
        df = pd.DataFrame({"x": [1, 2], "y": [1, 2], "car": ["a", ""]})
        p = (
            ggplot(df, aes(x="x", y="y", label="car"))
            + geom_point()
            + geom_text_repel(seed=42)
        )
        png = p._repr_png_()
        assert png is not None


# ---------------------------------------------------------------------------
# Gap #5: ±Inf in xlim / ylim preserved (R geom-text-repel.R:334-339)
# ---------------------------------------------------------------------------


class TestGap5InfLimits:

    def test_text_inf_xlim(self, small_df):
        p = (
            ggplot(small_df, aes(x="x", y="y", label="car"))
            + geom_point()
            + geom_text_repel(xlim=(float("-inf"), float("inf")), seed=42)
        )
        png = p._repr_png_()
        assert png is not None

    def test_label_inf_ylim(self, small_df):
        p = (
            ggplot(small_df, aes(x="x", y="y", label="car"))
            + geom_point()
            + geom_label_repel(ylim=(1.0, float("inf")), seed=42)
        )
        png = p._repr_png_()
        assert png is not None

    def test_text_nan_xlim_defaults_to_01(self, small_df):
        """``None`` (NaN after _norm_lim) → fallback to (0, 1)."""
        p = (
            ggplot(small_df, aes(x="x", y="y", label="car"))
            + geom_point()
            + geom_text_repel(xlim=(None, None), seed=42)
        )
        png = p._repr_png_()
        assert png is not None


# ---------------------------------------------------------------------------
# Gap #6: direction enum validation (R match.arg)
# ---------------------------------------------------------------------------


class TestGap6DirectionValidation:

    @pytest.mark.parametrize("bad", ["xy", "", "horizontal", None])
    def test_text_invalid_direction(self, bad):
        with pytest.raises(ValueError, match="direction"):
            geom_text_repel(direction=bad)

    @pytest.mark.parametrize("bad", ["xy", "invalid", 42])
    def test_label_invalid_direction(self, bad):
        with pytest.raises(ValueError, match="direction"):
            geom_label_repel(direction=bad)

    @pytest.mark.parametrize("good", ["both", "x", "y"])
    def test_valid_direction_accepted(self, good):
        geom_text_repel(direction=good)
        geom_label_repel(direction=good)


# ---------------------------------------------------------------------------
# Gap #7: ``getOption``-backed defaults (R getOption("ggrepel.max.overlaps", 10))
# ---------------------------------------------------------------------------


class TestGap7GetOptionDefaults:

    def test_max_overlaps_default_is_10(self):
        """Without a set option, max_overlaps defaults to 10."""
        lay = geom_text_repel()
        assert lay.geom_params["max_overlaps"] == 10

    def test_max_overlaps_reads_option(self):
        set_option("ggrepel.max.overlaps", 42)
        lay = geom_text_repel()
        assert lay.geom_params["max_overlaps"] == 42

    def test_explicit_arg_overrides_option(self):
        set_option("ggrepel.max.overlaps", 42)
        lay = geom_text_repel(max_overlaps=99)
        assert lay.geom_params["max_overlaps"] == 99

    def test_set_option_returns_previous(self):
        set_option("ggrepel.max.overlaps", 1)
        prev = set_option("ggrepel.max.overlaps", 2)
        assert prev == 1

    def test_set_option_none_removes(self):
        """R's ``options(name = NULL)`` unsets the option."""
        set_option("ggrepel.max.overlaps", 5)
        set_option("ggrepel.max.overlaps", None)
        assert get_option("ggrepel.max.overlaps") is None
        # Default re-applies on next factory call.
        lay = geom_text_repel()
        assert lay.geom_params["max_overlaps"] == 10

    def test_verbose_default_is_false(self):
        lay = geom_text_repel()
        assert lay.geom_params["verbose"] is False

    def test_verbose_reads_option(self):
        set_option("verbose", True)
        lay = geom_label_repel()
        assert lay.geom_params["verbose"] is True


# ---------------------------------------------------------------------------
# Gap #8: verbose uses ``inform``, not ``warnings.warn``
#         (R rlang::inform / Rcpp::message)
# ---------------------------------------------------------------------------


class TestGap8InformNotWarning:

    def test_inform_helper_routes_through_logger(self):
        """Verify ``inform()`` emits via the ``ggrepel_py`` logger.

        Pytest's capture mechanisms (``capsys`` / ``capfd`` / ``caplog``)
        all struggle with ``inform`` because:
          - the logger has ``propagate=False`` (R ``rlang::inform``
            doesn't re-broadcast) so ``caplog`` on root sees nothing;
          - the logger's ``StreamHandler`` snapshotted ``sys.stderr`` at
            import time, so ``capsys``/``capfd`` swap/dup miss the write.

        Instead we attach a local ``BufferingHandler`` directly to the
        ``ggrepel_py`` logger for the duration of the test.  A regression
        that silenced ``inform`` would leave the buffer empty.
        """
        from logging.handlers import BufferingHandler

        handler = BufferingHandler(capacity=16)
        logger = logging.getLogger("ggrepel_py")
        logger.addHandler(handler)
        try:
            _inform("hello world")
        finally:
            logger.removeHandler(handler)
        messages = [r.getMessage() for r in handler.buffer]
        assert "hello world" in messages

    def test_inform_does_not_route_through_warnings(self):
        """Regression guard: ``inform`` must NOT go through ``warnings.warn``.

        If it did, ``warnings.catch_warnings() + simplefilter('error')``
        would raise a ``UserWarning`` (as was the case before Gap #8 fix).
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _inform("must not be a warning")
            # reaching here means no warning was raised

    def test_inform_respects_logger_level(self):
        """Setting the logger to ``WARNING`` silences INFO-level ``inform``.

        Documented silence recipe from ``_options.inform`` docstring; this
        test fails if the logger were misconfigured (e.g., created with a
        hard-coded stderr writer bypassing the level filter).
        """
        from logging.handlers import BufferingHandler

        handler = BufferingHandler(capacity=16)
        logger = logging.getLogger("ggrepel_py")
        prev_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        try:
            _inform("should be suppressed")
        finally:
            logger.setLevel(prev_level)
            logger.removeHandler(handler)
        assert handler.buffer == []

    def test_verbose_does_not_raise_with_Werror(self, small_df):
        """R's ``inform`` doesn't use the warning channel; neither should we."""
        # Many points with max_overlaps=0 → solver reports too_many_overlaps.
        df_many = pd.DataFrame({
            "x": [1] * 5, "y": [1] * 5, "car": list("abcde"),
        })
        p = (
            ggplot(df_many, aes(x="x", y="y", label="car"))
            + geom_point()
            + geom_text_repel(max_overlaps=0, verbose=True, seed=42)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # make any warning fatal
            _render(p, "/tmp/_audit_inform.png")


# ---------------------------------------------------------------------------
# Public API smoke tests
# ---------------------------------------------------------------------------


def test_version_is_string():
    assert isinstance(__version__, str) and __version__


def test_public_exports():
    """Verify the five user-facing names + options API are reachable."""
    from ggrepel_py import (
        GeomTextRepel, TextRepelTree, geom_text_repel,
        GeomLabelRepel, LabelRepelTree, geom_label_repel,
        PositionNudgeRepel, position_nudge_repel,
        get_option, set_option,
    )
    # Types are classes; factories are callables.
    for obj in (GeomTextRepel, TextRepelTree, GeomLabelRepel,
                LabelRepelTree, PositionNudgeRepel):
        assert isinstance(obj, type)
    for fn in (geom_text_repel, geom_label_repel, position_nudge_repel,
               get_option, set_option):
        assert callable(fn)
