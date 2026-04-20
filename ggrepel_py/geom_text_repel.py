"""``geom_text_repel`` — text layer with force-directed anti-overlap layout.

Port of ``ggrepel/R/geom-text-repel.R``. Three public pieces:

* :class:`GeomTextRepel` — ``ggplot2_py.GeomText`` subclass whose ``draw_panel``
  returns a :class:`TextRepelTree`.
* :class:`TextRepelTree` — ``grid_py.GTree`` subclass whose ``make_content``
  runs the ``repel_boxes2`` solver and materialises the text/segment children
  just before rendering.
* :func:`geom_text_repel` — user-facing constructor that mirrors R's
  ``geom_text_repel()``.

R aesthetics with ``.`` in their names (``segment.colour``, ``bg.colour``, ...)
are renamed to their ``snake_case`` equivalents (``segment_colour``,
``bg_colour``). Argument names with ``.`` (``box.padding`` →  ``box_padding``,
``na.rm`` → ``na_rm``, ``max.time`` → ``max_time``, ``min.segment.length`` →
``min_segment_length``) are renamed the same way.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import grid_py as grid
from grid_py import (
    GList,
    GTree,
    Gpar,
    Unit,
    convert_height,
    convert_width,
    curve_grob,
    grob_x,
    grob_y,
    null_grob,
    string_height,
    string_width,
    text_grob,
)

from ggplot2_py.aes import Mapping
from ggplot2_py.geom import GeomText
from ggplot2_py.layer import layer as _layer
from scales import alpha as _alpha

from ggrepel_py import _repel
from ggrepel_py._utilities import (
    PT,
    compute_just,
    ggname,
    not_empty,
    null_default,
    to_unit,
)
from ggrepel_py.position_nudge_repel import position_nudge_repel

__all__ = ["GeomTextRepel", "TextRepelTree", "geom_text_repel"]


_STROKE = 0.96  # ggplot2's .stroke constant (see ggplot2/R/utilities.R)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _coord_transform(coord: Any, data: pd.DataFrame, panel_params: Any) -> pd.DataFrame:
    """Run ``coord.transform`` when available, else pass through."""
    if coord is not None and hasattr(coord, "transform"):
        return coord.transform(data, panel_params)
    return data


def _as_float_array(v: Any, n: int, default: float) -> np.ndarray:
    """Broadcast *v* to a length-*n* float array, substituting *default* for NaN."""
    if v is None:
        return np.full(n, default, dtype=float)
    arr = np.asarray(v, dtype=float).ravel()
    if arr.size == 1:
        arr = np.full(n, float(arr[0]))
    elif arr.size != n:
        arr = np.resize(arr, n)
    arr = np.where(np.isnan(arr), default, arr)
    return arr


def _convert_width_native(u: Any) -> float:
    """Convert a single unit / number to native (x) units as a scalar."""
    if not grid.is_unit(u):
        u = Unit(float(u), "native")
    v = convert_width(u, "native", valueOnly=True)
    return float(np.asarray(v).ravel()[0])


def _convert_height_native(u: Any) -> float:
    """Convert a single unit / number to native (y) units as a scalar."""
    if not grid.is_unit(u):
        u = Unit(float(u), "native")
    v = convert_height(u, "native", valueOnly=True)
    return float(np.asarray(v).ravel()[0])


# ---------------------------------------------------------------------------
# shadowtext port
# ---------------------------------------------------------------------------

def _shadowtext_grobs(
    label: str,
    x: Any,
    y: Any,
    rot: float,
    hjust: float,
    vjust: float,
    gp: Gpar,
    name: str,
    bg_colour: Any,
    bg_r: float,
) -> List[Any]:
    """Return a list of grobs: optional halo textGrobs followed by the main text grob.

    Mirrors ``shadowtextGrob`` in ``geom-text-repel.R``.
    """
    upper = text_grob(
        label=label,
        x=x,
        y=y,
        hjust=hjust,
        vjust=vjust,
        rot=rot,
        default_units="native",
        name=name,
        gp=gp,
    )
    if bg_colour is None or (isinstance(bg_colour, float) and math.isnan(bg_colour)):
        return [upper]

    # Render a halo of 16 offset textGrobs underneath.
    halo_gp = Gpar(**{**gp.params, "col": bg_colour}) if hasattr(gp, "params") else Gpar(col=bg_colour)
    try:
        # Prefer merging the original Gpar fields so font face / family carry over.
        merged = dict(getattr(gp, "_params", {}) or {})
        merged["col"] = bg_colour
        halo_gp = Gpar(**merged)
    except Exception:
        halo_gp = Gpar(col=bg_colour)

    char = "X"
    thetas = np.linspace(math.pi / 8, 2 * math.pi, 16)
    r = float(bg_r)

    # Ensure x/y are Units so we can add strheight offsets.
    if not grid.is_unit(x):
        x = Unit(float(x), "native")
    if not grid.is_unit(y):
        y = Unit(float(y), "native")

    grobs: List[Any] = []
    for i, t in enumerate(thetas):
        dx = Unit(float(math.cos(t) * r), "strheight", data=char)
        dy = Unit(float(math.sin(t) * r), "strheight", data=char)
        grobs.append(text_grob(
            label=label,
            x=x + dx,
            y=y + dy,
            hjust=hjust,
            vjust=vjust,
            rot=rot,
            default_units="native",
            name=f"{name}-shadowtext{i}",
            gp=halo_gp,
        ))
    grobs.append(upper)
    return grobs


# ---------------------------------------------------------------------------
# makeTextRepelGrobs port
# ---------------------------------------------------------------------------

def _make_repel_grobs(
    i: int,
    label: str,
    x: Any,
    y: Any,
    x_orig: float,
    y_orig: float,
    rot: float,
    box_padding: Any,
    point_size: float,
    point_padding: float,
    segment_curvature: float,
    segment_angle: float,
    segment_ncp: int,
    segment_shape: float,
    segment_square: bool,
    segment_square_shape: float,
    segment_inflect: bool,
    text_gp: Gpar,
    segment_gp: Gpar,
    arrow: Any,
    min_segment_length: Any,
    hjust: float,
    vjust: float,
    bg_colour: Any,
    bg_r: float,
    dim: Tuple[float, float],
) -> List[Any]:
    """Build the text grob (with optional halo) and, when warranted, a leader-line
    curve grob for label *i*.

    Port of ``makeTextRepelGrobs`` in ``geom-text-repel.R``.
    """
    if not grid.is_unit(x):
        x = Unit(float(x), "native")
    if not grid.is_unit(y):
        y = Unit(float(y), "native")

    rot = float(rot) % 360
    rot_rad = rot * math.pi / 180.0

    string_h = convert_height(string_height(label), "char")
    string_w = convert_width(string_width(label), "char")

    # x_adj = x - cos * width * (0.5 - hjust) + sin * height * (0.5 - vjust)
    x_adj = (
        x
        + Unit(-math.cos(rot_rad) * (0.5 - hjust), "char") * string_w.values[0]
        + Unit(math.sin(rot_rad) * (0.5 - vjust), "char") * string_h.values[0]
    )
    y_adj = (
        y
        + Unit(-math.cos(rot_rad) * (0.5 - vjust), "char") * string_h.values[0]
        + Unit(-math.sin(rot_rad) * (0.5 - hjust), "char") * string_w.values[0]
    )

    text_name = f"textrepelgrob{i}"
    grobs = _shadowtext_grobs(
        label=label,
        x=x_adj,
        y=y_adj,
        rot=rot,
        hjust=hjust,
        vjust=vjust,
        gp=text_gp,
        name=text_name,
        bg_colour=bg_colour,
        bg_r=bg_r,
    )

    tg = grobs[-1]  # the main (non-halo) text grob

    x1 = float(convert_width(grob_x(tg, "west"), "native", valueOnly=True)[0])
    x2 = float(convert_width(grob_x(tg, "east"), "native", valueOnly=True)[0])
    y1 = float(convert_height(grob_y(tg, "south"), "native", valueOnly=True)[0])
    y2 = float(convert_height(grob_y(tg, "north"), "native", valueOnly=True)[0])

    point_pos = np.array([float(x_orig), float(y_orig)])

    extra_pad_x = _convert_width_native(Unit(0.25, "lines")) / 2.0
    extra_pad_y = _convert_height_native(Unit(0.25, "lines")) / 2.0
    text_box = np.array([
        x1 - extra_pad_x,
        y1 - extra_pad_y,
        x2 + extra_pad_x,
        y2 + extra_pad_y,
    ])

    intersection = _repel.select_line_connection(point_pos, text_box)

    inside = (
        text_box[0] <= point_pos[0] <= text_box[2]
        and text_box[1] <= point_pos[1] <= text_box[3]
    )

    dim_arr = np.asarray(dim, dtype=float)
    point_int = _repel.intersect_line_circle(
        intersection * dim_arr, point_pos * dim_arr, point_size + point_padding
    ) / dim_arr

    dx = abs(intersection[0] - point_int[0])
    dy = abs(intersection[1] - point_int[1])
    d = math.sqrt(dx * dx + dy * dy)

    if d > 0:
        mx = _convert_width_native(min_segment_length) if min_segment_length is not None else float("nan")
        my = _convert_height_native(min_segment_length) if min_segment_length is not None else float("nan")
        min_seg_len: float = math.sqrt((mx * dx / d) ** 2 + (my * dy / d) ** 2)
    else:
        min_seg_len = float("nan")

    draw_segment = (
        not inside
        and d > 0
        and (not math.isnan(min_seg_len) and _repel.euclid(intersection, point_int) > min_seg_len)
        and _repel.euclid(intersection, point_int) < _repel.euclid(intersection, point_pos)
        and _repel.euclid(intersection * dim_arr, point_pos * dim_arr) > point_size
        and _repel.euclid(intersection, point_pos) > _repel.euclid(point_int, point_pos)
    )

    if draw_segment:
        seg_name = f"segmentrepelgrob{i}"
        s = curve_grob(
            x1=float(intersection[0]),
            y1=float(intersection[1]),
            x2=float(point_int[0]),
            y2=float(point_int[1]),
            default_units="native",
            curvature=float(segment_curvature),
            angle=float(segment_angle),
            ncp=int(segment_ncp),
            shape=float(segment_shape),
            square=bool(segment_square),
            squareShape=float(segment_square_shape),
            inflect=bool(segment_inflect),
            arrow=arrow,
            gp=segment_gp,
            name=seg_name,
        )
        grobs.append(s)

    return grobs


# ---------------------------------------------------------------------------
# TextRepelTree: GTree subclass that runs the solver at make_content time
# ---------------------------------------------------------------------------

class TextRepelTree(GTree):
    """Grid tree whose ``make_content`` hook lays out labels via ``repel_boxes2``.

    Port of the ``textrepeltree`` S3 class dispatched through
    ``makeContent.textrepeltree`` in ``geom-text-repel.R``.
    """

    def __init__(
        self,
        *,
        data: pd.DataFrame,
        lab: Any,
        limits: pd.DataFrame,
        box_padding: Any,
        point_padding: Any,
        min_segment_length: Any,
        arrow: Any,
        force: float,
        force_pull: float,
        max_time: float,
        max_iter: int,
        max_overlaps: float,
        direction: str,
        seed: Any,
        verbose: bool,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            children=None,
            name=name,
            _grid_class="textrepeltree",
        )
        # Attributes used by make_content
        self.data = data
        self.lab = list(lab)
        self.limits = limits
        self.box_padding = box_padding
        self.point_padding = point_padding
        self.min_segment_length = min_segment_length
        self.arrow = arrow
        self.force = float(force)
        self.force_pull = float(force_pull)
        self.max_time = float(max_time)
        self.max_iter = int(max_iter) if not math.isinf(max_iter) else 10**9
        self.max_overlaps = float(max_overlaps)
        self.direction = str(direction)
        self.seed = seed
        self.verbose = bool(verbose)

    # ---------------------------------------------------------------- helpers
    def _row_text_gp(self, row: Any) -> Gpar:
        return Gpar(
            col=_alpha(row.get("colour", "black"), row.get("alpha")),
            fontsize=float(row.get("size", 3.88)) * PT,
            fontfamily=row.get("family", ""),
            fontface=row.get("fontface", 1),
            lineheight=row.get("lineheight", 1.2),
        )

    def _row_segment_gp(self, row: Any) -> Gpar:
        seg_colour = null_default(row.get("segment_colour"), row.get("colour", "black"))
        seg_alpha = null_default(row.get("segment_alpha"), row.get("alpha"))
        arrow_fill = null_default(row.get("arrow_fill"), seg_colour)
        return Gpar(
            col=_alpha(seg_colour, seg_alpha),
            lwd=float(row.get("segment_size", 0.5)) * PT,
            lty=null_default(row.get("segment_linetype"), 1),
            fill=_alpha(arrow_fill, seg_alpha),
        )

    # ---------------------------------------------------------------- main hook
    def make_content(self) -> "TextRepelTree":
        data = self.data.reset_index(drop=True).copy()
        lab = list(self.lab)
        n = len(data)
        if n == 0:
            self.set_children(GList())
            return self

        box_padding_x = _convert_width_native(self.box_padding)
        box_padding_y = _convert_height_native(self.box_padding)

        point_padding = self.point_padding
        if point_padding is None or (isinstance(point_padding, float) and math.isnan(point_padding)):
            point_padding = Unit(0, "lines")

        # Re-order rows so valid labels come first (matching R).
        valid_mask = not_empty(lab)
        valid_idx = list(np.where(valid_mask)[0])
        invalid_idx = list(np.where(~valid_mask)[0])
        order = valid_idx + invalid_idx
        data = data.iloc[order].reset_index(drop=True)
        lab = [lab[i] for i in order]
        n_valid = len(valid_idx)

        if n_valid == 0:
            self.set_children(GList())
            return self

        # Build per-label bounding boxes (x1, y1, x2, y2) in native coords.
        angle_vals = _as_float_array(data.get("angle", 0), n, 0.0)
        size_vals = _as_float_array(data.get("size", 3.88), n, 3.88)
        hjust_vals = _as_float_array(data.get("hjust", 0.5), n, 0.5)
        vjust_vals = _as_float_array(data.get("vjust", 0.5), n, 0.5)
        nudge_x_vals = _as_float_array(data.get("nudge_x", 0.0), n, 0.0)
        nudge_y_vals = _as_float_array(data.get("nudge_y", 0.0), n, 0.0)
        family_vals = [str(v) if v is not None else "" for v in data.get("family", [""] * n)]
        fontface_vals = list(data.get("fontface", [1] * n))
        lineheight_vals = _as_float_array(data.get("lineheight", 1.2), n, 1.2)

        boxes = np.zeros((n_valid, 4), dtype=float)
        for i in range(n_valid):
            tg = text_grob(
                label=str(lab[i]),
                x=float(data.iloc[i]["x"]),
                y=float(data.iloc[i]["y"]),
                default_units="native",
                rot=float(angle_vals[i]),
                hjust=float(hjust_vals[i]),
                vjust=float(vjust_vals[i]),
                gp=Gpar(
                    fontsize=float(size_vals[i]) * PT,
                    fontfamily=family_vals[i],
                    fontface=fontface_vals[i],
                    lineheight=float(lineheight_vals[i]),
                ),
            )
            x1 = float(convert_width(grob_x(tg, "west"), "native", valueOnly=True)[0])
            x2 = float(convert_width(grob_x(tg, "east"), "native", valueOnly=True)[0])
            y1 = float(convert_height(grob_y(tg, "south"), "native", valueOnly=True)[0])
            y2 = float(convert_height(grob_y(tg, "north"), "native", valueOnly=True)[0])
            boxes[i] = [
                x1 - box_padding_x + nudge_x_vals[i],
                y1 - box_padding_y + nudge_y_vals[i],
                x2 + box_padding_x + nudge_x_vals[i],
                y2 + box_padding_y + nudge_y_vals[i],
            ]

        # Resolve the seed: NaN/None → pick a random draw so runs differ;
        # otherwise use as-is.
        seed = self.seed
        if seed is None or (isinstance(seed, float) and math.isnan(seed)):
            seed = int(np.random.default_rng().integers(1, 2**31 - 1))
        seed = int(seed)

        # Magic-number scaling lifted from the R source.
        p_width = _convert_width_native(Unit(1, "npc")) if False else float(
            convert_width(Unit(1, "npc"), "inch", valueOnly=True)[0]
        )
        p_height = float(convert_height(Unit(1, "npc"), "inch", valueOnly=True)[0])
        p_ratio = p_width / p_height if p_height > 0 else 1.0
        if p_ratio > 1:
            p_ratio = p_ratio ** (1 / (1.15 * p_ratio))

        point_size_series = _as_float_array(data.get("point_size", 1.0), n, 0.0)
        # Every data point is a repulsion source for every text label,
        # including rows whose label is "" (no box, but still a point).
        point_size_arr = (
            p_ratio
            * np.array([_convert_width_native(to_unit(v)) for v in point_size_series])
            / 13.0
        )
        point_padding_native = (
            p_ratio * _convert_width_native(to_unit(point_padding)) / 13.0
        )

        lx = np.asarray(self.limits["x"].to_numpy(), dtype=float)
        ly = np.asarray(self.limits["y"].to_numpy(), dtype=float)
        xlim = np.array([np.nanmin(lx), np.nanmax(lx)])
        ylim = np.array([np.nanmin(ly), np.nanmax(ly)])

        data_points = data[["x", "y"]].to_numpy(dtype=float)

        repel = _repel.repel_boxes2(
            data_points=data_points,
            point_size=point_size_arr,
            point_padding_x=float(point_padding_native),
            point_padding_y=float(point_padding_native),
            boxes=boxes,
            xlim=xlim,
            ylim=ylim,
            hjust=hjust_vals,
            vjust=vjust_vals,
            force_push=float(self.force) * 1e-6,
            force_pull=float(self.force_pull) * 1e-2,
            max_time=float(self.max_time),
            max_overlaps=float(self.max_overlaps),
            max_iter=int(self.max_iter),
            direction=self.direction,
            verbose=1 if self.verbose else 0,
            seed=seed,
        )

        too_many = np.asarray(repel["too_many_overlaps"], dtype=bool)
        if self.verbose and too_many.any():
            import warnings
            n_skip = int(too_many.sum())
            warnings.warn(
                f"ggrepel: {n_skip} unlabeled data point(s) (too many overlaps). "
                "Consider increasing `max_overlaps`."
            )
        if too_many.all():
            self.set_children(GList())
            return self

        # Unit conversions for the point-size / point-padding the segment path uses.
        point_size_cm = (
            _as_float_array(data.get("point_size", 1.0), n, 0.0) * PT / _STROKE / 20.0
        )
        point_padding_cm = float(convert_width(to_unit(point_padding), "cm", valueOnly=True)[0])
        width_cm = float(convert_width(Unit(1, "npc"), "cm", valueOnly=True)[0])
        height_cm = float(convert_height(Unit(1, "npc"), "cm", valueOnly=True)[0])

        x_solved = np.asarray(repel["x"], dtype=float)
        y_solved = np.asarray(repel["y"], dtype=float)

        # Per-aesthetic columns the segment/arrow grobs need.
        arrow_fill_vals = data.get("arrow_fill", [None] * n)
        bg_colour_vals = data.get("bg_colour", [None] * n)
        bg_r_vals = _as_float_array(data.get("bg_r", 0.1), n, 0.1)
        seg_curv = _as_float_array(data.get("segment_curvature", 0.0), n, 0.0)
        seg_ang = _as_float_array(data.get("segment_angle", 90.0), n, 90.0)
        seg_ncp = _as_float_array(data.get("segment_ncp", 1), n, 1).astype(int)
        seg_shape = _as_float_array(data.get("segment_shape", 0.5), n, 0.5)
        seg_square = list(data.get("segment_square", [True] * n))
        seg_sq_shape = _as_float_array(data.get("segment_square_shape", 1.0), n, 1.0)
        seg_inflect = list(data.get("segment_inflect", [False] * n))

        all_grobs: List[Any] = []
        for i in range(n_valid):
            if too_many[i]:
                continue
            row = data.iloc[i]

            all_grobs.extend(_make_repel_grobs(
                i=i,
                label=str(lab[i]),
                x=Unit(float(x_solved[i]), "native"),
                y=Unit(float(y_solved[i]), "native"),
                x_orig=float(row["x"]),
                y_orig=float(row["y"]),
                rot=float(angle_vals[i]),
                box_padding=self.box_padding,
                point_size=float(point_size_cm[i]),
                point_padding=float(point_padding_cm),
                segment_curvature=float(seg_curv[i]),
                segment_angle=float(seg_ang[i]),
                segment_ncp=int(seg_ncp[i]),
                segment_shape=float(seg_shape[i]),
                segment_square=bool(seg_square[i]),
                segment_square_shape=float(seg_sq_shape[i]),
                segment_inflect=bool(seg_inflect[i]),
                text_gp=Gpar(
                    col=_alpha(row.get("colour", "black"), row.get("alpha")),
                    fontsize=float(size_vals[i]) * PT,
                    fontfamily=family_vals[i],
                    fontface=fontface_vals[i],
                    lineheight=float(lineheight_vals[i]),
                ),
                segment_gp=self._row_segment_gp(
                    {
                        "colour": row.get("colour", "black"),
                        "alpha": row.get("alpha"),
                        "segment_colour": row.get("segment_colour"),
                        "segment_alpha": row.get("segment_alpha"),
                        "segment_size": row.get("segment_size", 0.5),
                        "segment_linetype": row.get("segment_linetype", 1),
                        "arrow_fill": arrow_fill_vals.iloc[i] if hasattr(arrow_fill_vals, "iloc") else arrow_fill_vals[i],
                    }
                ),
                arrow=self.arrow,
                min_segment_length=self.min_segment_length,
                hjust=float(hjust_vals[i]),
                vjust=float(vjust_vals[i]),
                bg_colour=bg_colour_vals.iloc[i] if hasattr(bg_colour_vals, "iloc") else bg_colour_vals[i],
                bg_r=float(bg_r_vals[i]),
                dim=(width_cm, height_cm),
            ))

        # R sorts grobs so segments render before text.
        all_grobs.sort(key=lambda g: 0 if str(getattr(g, "name", "")).startswith("segment") else 1)
        self.set_children(GList(*all_grobs))
        return self


# ---------------------------------------------------------------------------
# GeomTextRepel ggproto
# ---------------------------------------------------------------------------


class GeomTextRepel(GeomText):
    """Text geom that avoids overlaps by repelling labels away from each other
    and from the data points they annotate. Port of ``GeomTextRepel``.
    """

    required_aes: Tuple[str, ...] = ("x", "y", "label")
    default_aes: Mapping = Mapping(
        colour="black",
        size=3.88,
        angle=0,
        alpha=None,
        family="",
        fontface=1,
        lineheight=1.2,
        hjust=0.5,
        vjust=0.5,
        point_size=1,
        segment_linetype=1,
        segment_colour=None,
        segment_size=0.5,
        segment_alpha=None,
        segment_curvature=0,
        segment_angle=90,
        segment_ncp=1,
        segment_shape=0.5,
        segment_square=True,
        segment_square_shape=1,
        segment_inflect=False,
        segment_debug=False,
        arrow_fill=None,
        bg_colour=None,
        bg_r=0.1,
    )

    def draw_panel(  # type: ignore[override]
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        parse: bool = False,
        na_rm: bool = False,
        box_padding: Any = 0.25,
        point_padding: Any = 1e-6,
        min_segment_length: Any = 0.5,
        arrow: Any = None,
        force: float = 1.0,
        force_pull: float = 1.0,
        max_time: float = 0.5,
        max_iter: int = 10000,
        max_overlaps: float = 10,
        nudge_x: Any = 0,
        nudge_y: Any = 0,
        xlim: Any = (None, None),
        ylim: Any = (None, None),
        direction: str = "both",
        seed: Any = None,
        verbose: bool = False,
        **_: Any,
    ) -> Any:
        if data is None or len(data) == 0:
            return null_grob()
        data = data.reset_index(drop=True).copy()

        # Rename x/y/x_orig/y_orig/nudge_x/nudge_y per the R draw_panel rules.
        for dim in ("x", "y"):
            orig_col = f"{dim}_orig"
            nudge_col = f"nudge_{dim}"
            if nudge_col not in data.columns:
                data[nudge_col] = data[dim]
                if orig_col in data.columns:
                    data[dim] = data[orig_col]
                    del data[orig_col]
            else:
                if orig_col in data.columns:
                    data[nudge_col] = data[dim]
                    data[dim] = data[orig_col]
                    del data[orig_col]
                elif (data[nudge_col] == 0).all():
                    data[nudge_col] = data[dim]

        # Transform nudges and raw data to the panel scales.
        nudges = pd.DataFrame({"x": data["nudge_x"].to_numpy(),
                               "y": data["nudge_y"].to_numpy()})
        nudges = _coord_transform(coord, nudges, panel_params)
        data = _coord_transform(coord, data, panel_params)
        data["nudge_x"] = nudges["x"].to_numpy() - data["x"].to_numpy()
        data["nudge_y"] = nudges["y"].to_numpy() - data["y"].to_numpy()

        # Build limits DataFrame; NaN/None → default range (0, 1).
        def _norm_lim(lim: Any) -> np.ndarray:
            if lim is None:
                return np.array([np.nan, np.nan])
            arr = np.asarray(
                [np.nan if v is None else v for v in lim], dtype=float
            ).ravel()
            if arr.size < 2:
                arr = np.resize(arr, 2)
            return arr[:2]

        xlim_arr = _norm_lim(xlim)
        ylim_arr = _norm_lim(ylim)
        xlim_na = np.isnan(xlim_arr)
        ylim_na = np.isnan(ylim_arr)
        limits_df = pd.DataFrame({"x": xlim_arr, "y": ylim_arr})
        limits_df = _coord_transform(coord, limits_df, panel_params)
        # Fill NA slots with defaults (0, 1).
        limits_df.loc[xlim_na, "x"] = np.array([0.0, 1.0])[xlim_na]
        limits_df.loc[ylim_na, "y"] = np.array([0.0, 1.0])[ylim_na]

        # Resolve character hjust/vjust via compute_just.
        if "vjust" in data.columns and data["vjust"].dtype == object:
            data["vjust"] = compute_just(
                data["vjust"].astype(str).to_numpy(),
                data["y"].to_numpy(),
                data["x"].to_numpy(),
                _as_float_array(data.get("angle", 0), len(data), 0.0),
            )
        if "hjust" in data.columns and data["hjust"].dtype == object:
            data["hjust"] = compute_just(
                data["hjust"].astype(str).to_numpy(),
                data["x"].to_numpy(),
                data["y"].to_numpy(),
                _as_float_array(data.get("angle", 0), len(data), 0.0),
            )

        tree = TextRepelTree(
            data=data,
            lab=data["label"].tolist(),
            limits=limits_df,
            box_padding=to_unit(box_padding),
            point_padding=to_unit(point_padding),
            min_segment_length=to_unit(min_segment_length),
            arrow=arrow,
            force=force,
            force_pull=force_pull,
            max_time=max_time,
            max_iter=max_iter,
            max_overlaps=max_overlaps,
            direction=direction,
            seed=seed,
            verbose=verbose,
        )
        return ggname("geom_text_repel", tree)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def geom_text_repel(
    mapping: Any = None,
    data: Any = None,
    stat: str = "identity",
    position: Any = "identity",
    parse: bool = False,
    *,
    box_padding: Any = 0.25,
    point_padding: Any = 1e-6,
    min_segment_length: Any = 0.5,
    arrow: Any = None,
    force: float = 1.0,
    force_pull: float = 1.0,
    max_time: float = 0.5,
    max_iter: int = 10000,
    max_overlaps: float = 10,
    nudge_x: Any = 0,
    nudge_y: Any = 0,
    xlim: Any = (None, None),
    ylim: Any = (None, None),
    na_rm: bool = False,
    show_legend: Any = None,
    direction: str = "both",
    seed: Any = None,
    verbose: bool = False,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Repulsive text layer. Port of R ``geom_text_repel()``.

    Parameters
    ----------
    mapping, data, stat, position
        Same as :func:`ggplot2_py.geom_text`.
    parse
        If ``True``, label strings are parsed as plotmath expressions (currently
        treated as opaque strings; rendering support is backend-dependent).
    box_padding, point_padding, min_segment_length
        Padding / minimum-segment-length values. Scalars use ``"lines"`` as the
        default unit; pass a :class:`grid_py.Unit` for other units.
    arrow
        Optional :class:`grid_py.Arrow` for the leader line.
    force, force_pull
        Repulsion between labels and attraction toward the data point. Defaults
        1.0 each.
    max_time, max_iter, max_overlaps
        Solver stopping criteria.
    nudge_x, nudge_y
        Converted to ``position_nudge_repel(nudge_x, nudge_y)`` when *position*
        is left at its default.
    xlim, ylim
        Two-element ``(low, high)`` tuples (``None`` entries default to the
        plot range).
    direction
        ``"both"`` / ``"x"`` / ``"y"`` — axes along which labels may move.
    seed
        Integer or ``None``. ``None`` / ``NaN`` picks a fresh random seed.
    verbose
        If ``True``, emit a warning when labels are dropped due to ``max_overlaps``.
    """
    if (nudge_x != 0 or nudge_y != 0) and position != "identity":
        raise ValueError(
            "Both `position` and `nudge_x`/`nudge_y` are supplied. "
            "Use only one approach to adjust label positions."
        )
    if nudge_x != 0 or nudge_y != 0:
        position = position_nudge_repel(nudge_x, nudge_y)

    params: Dict[str, Any] = {
        "parse": parse,
        "na_rm": na_rm,
        "box_padding": to_unit(box_padding),
        "point_padding": to_unit(point_padding),
        "min_segment_length": to_unit(min_segment_length),
        "arrow": arrow,
        "force": force,
        "force_pull": force_pull,
        "max_time": max_time,
        "max_iter": max_iter,
        "max_overlaps": max_overlaps,
        "nudge_x": nudge_x,
        "nudge_y": nudge_y,
        "xlim": xlim,
        "ylim": ylim,
        "direction": direction,
        "seed": seed,
        "verbose": verbose,
    }
    params.update(kwargs)

    return _layer(
        geom=GeomTextRepel,
        stat=stat,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params=params,
    )
