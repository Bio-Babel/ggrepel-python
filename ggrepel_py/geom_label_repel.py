"""``geom_label_repel`` — labelled-box layer with force-directed layout.

Port of ``ggrepel/R/geom-label-repel.R``. Shares the repulsion engine and
segment logic with :mod:`ggrepel_py.geom_text_repel` but wraps each label in a
rounded background rectangle (``roundrect_grob``). The public pieces are:

* :class:`GeomLabelRepel` — ``ggplot2_py.GeomLabel`` subclass whose
  ``draw_panel`` returns a :class:`LabelRepelTree`.
* :class:`LabelRepelTree` — ``grid_py.GTree`` subclass whose ``make_content``
  runs the ``repel_boxes2`` solver and materialises the text/rect/segment
  children just before rendering.
* :func:`geom_label_repel` — user-facing constructor that mirrors R's
  ``geom_label_repel()``.
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
    null_grob,
    roundrect_grob,
    text_grob,
)

from ggplot2_py.aes import Mapping
from ggplot2_py.geom import GeomLabel
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
from ggrepel_py._options import get_option as _get_option
from ggrepel_py.geom_text_repel import (
    _UNSET,
    _as_float_array,
    _convert_height_native,
    _convert_width_native,
)
from ggrepel_py.position_nudge_repel import position_nudge_repel

__all__ = ["GeomLabelRepel", "LabelRepelTree", "geom_label_repel"]


_STROKE = 0.96  # ggplot2's .stroke constant.


# ---------------------------------------------------------------------------
# makeLabelRepelGrobs port
# ---------------------------------------------------------------------------

def _make_label_grobs(
    i: int,
    label: str,
    x: Any,
    y: Any,
    x_orig: float,
    y_orig: float,
    box_width: float,
    box_height: float,
    box_padding: Any,
    label_padding: Any,
    label_r: Any,
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
    rect_gp: Gpar,
    segment_gp: Gpar,
    arrow: Any,
    min_segment_length: Any,
    hjust: float,
    vjust: float,
    dim: Tuple[float, float],
) -> Dict[str, Any]:
    """Build ``{"textbox": [rect, text], "segment": Optional[curve]}`` for label *i*.

    Port of ``makeLabelRepelGrobs`` in ``geom-label-repel.R``.
    """
    if not grid.is_unit(x):
        x = Unit(float(x), "native")
    if not grid.is_unit(y):
        y = Unit(float(y), "native")
    if not grid.is_unit(label_padding):
        label_padding = Unit(float(label_padding), "lines")

    box_w_u = Unit(float(box_width), "native") if not grid.is_unit(box_width) else box_width
    box_h_u = Unit(float(box_height), "native") if not grid.is_unit(box_height) else box_height

    # Text anchor: shift by (0.5 - hjust) and (0.5 - vjust) of the box
    # dimensions so that hjust/vjust behave as centred/left/right anchors.
    t = text_grob(
        label=label,
        x=x + box_w_u * float(-(0.5 - hjust)),
        y=y + box_h_u * float(-(0.5 - vjust)),
        hjust=float(hjust),
        vjust=float(vjust),
        gp=text_gp,
        name=f"textrepelgrob{i}",
    )

    # Box anchor: shift text-anchor by (0.5 - hjust/vjust) * label_padding.
    lp_native_x = _convert_width_native(label_padding)
    lp_native_y = _convert_height_native(label_padding)

    grob_x_val = x + box_w_u * float(-(0.5 - hjust)) + Unit(-(0.5 - hjust) * lp_native_x, "native")
    grob_y_val = y + box_h_u * float(-(0.5 - vjust)) + Unit(-(0.5 - vjust) * lp_native_y, "native")

    grob_w = box_w_u + Unit(2 * lp_native_x, "native")
    grob_h = box_h_u + Unit(2 * lp_native_y, "native")

    r_grob = roundrect_grob(
        x=grob_x_val,
        y=grob_y_val,
        default_units="native",
        width=grob_w,
        height=grob_h,
        just=(float(hjust), float(vjust)),
        r=label_r,
        gp=rect_gp,
        name=f"rectrepelgrob{i}",
    )

    # Bounding box edges in native for segment logic.
    gx_n = _convert_width_native(grob_x_val)
    gy_n = _convert_height_native(grob_y_val)
    gw_n = _convert_width_native(grob_w)
    gh_n = _convert_height_native(grob_h)
    x1 = gx_n - hjust * gw_n
    x2 = gx_n + (1 - hjust) * gw_n
    y1 = gy_n - vjust * gh_n
    y2 = gy_n + (1 - vjust) * gh_n
    text_box = np.array([x1, y1, x2, y2])

    point_pos = np.array([float(x_orig), float(y_orig)])
    intersection = _repel.select_line_connection(point_pos, text_box)

    inside = (
        text_box[0] <= point_pos[0] <= text_box[2]
        and text_box[1] <= point_pos[1] <= text_box[3]
    )

    dim_arr = np.asarray(dim, dtype=float)
    point_int = _repel.intersect_line_circle(
        intersection * dim_arr, point_pos * dim_arr, point_size + point_padding
    ) / dim_arr

    dx = abs(intersection[0] - point_pos[0])
    dy = abs(intersection[1] - point_pos[1])
    d = math.sqrt(dx * dx + dy * dy)

    if d > 0:
        mx = _convert_width_native(min_segment_length) if min_segment_length is not None else float("nan")
        my = _convert_height_native(min_segment_length) if min_segment_length is not None else float("nan")
        min_seg_len: float = math.sqrt((mx * dx / d) ** 2 + (my * dy / d) ** 2)
    else:
        min_seg_len = float("nan")

    out: Dict[str, Any] = {"textbox": [r_grob, t]}

    draw_segment = (
        not inside
        and d > 0
        and (not math.isnan(min_seg_len) and _repel.euclid(intersection, point_int) > min_seg_len)
        and _repel.euclid(intersection, point_int) < _repel.euclid(intersection, point_pos)
        and _repel.euclid(intersection * dim_arr, point_pos * dim_arr) > point_size
        and _repel.euclid(intersection, point_pos) > _repel.euclid(point_int, point_pos)
    )

    if draw_segment:
        out["segment"] = curve_grob(
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
            name=f"segmentrepelgrob{i}",
        )

    return out


# ---------------------------------------------------------------------------
# LabelRepelTree
# ---------------------------------------------------------------------------


class LabelRepelTree(GTree):
    """Grid tree for labelled boxes whose ``make_content`` hook runs
    ``repel_boxes2`` and emits rect + text + curve children in the R order.
    """

    def __init__(
        self,
        *,
        data: pd.DataFrame,
        lab: Any,
        limits: pd.DataFrame,
        box_padding: Any,
        label_padding: Any,
        point_padding: Any,
        label_r: Any,
        label_size: float,
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
            _grid_class="labelrepeltree",
        )
        self.data = data
        self.lab = list(lab)
        self.limits = limits
        self.box_padding = box_padding
        self.label_padding = label_padding
        self.point_padding = point_padding
        self.label_r = label_r
        self.label_size = float(label_size)
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

    # ------------------------------------------------------------------ main
    def make_content(self) -> "LabelRepelTree":
        data = self.data.reset_index(drop=True).copy()
        lab = list(self.lab)
        n = len(data)
        if n == 0:
            self.set_children(GList())
            return self

        # R uses npc here for the box-padding (makeContent.labelrepeltree:243-244).
        box_padding_x = float(convert_width(self.box_padding, "npc", valueOnly=True)[0])
        box_padding_y = float(convert_height(self.box_padding, "npc", valueOnly=True)[0])

        point_padding = self.point_padding
        if point_padding is None or (isinstance(point_padding, float) and math.isnan(point_padding)):
            point_padding = Unit(0, "lines")

        # Re-order valid-first.
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

        hjust_vals = _as_float_array(data.get("hjust", 0.5), n, 0.5)
        vjust_vals = _as_float_array(data.get("vjust", 0.5), n, 0.5)
        size_vals = _as_float_array(data.get("size", 3.88), n, 3.88)
        nudge_x_vals = _as_float_array(data.get("nudge_x", 0.0), n, 0.0)
        nudge_y_vals = _as_float_array(data.get("nudge_y", 0.0), n, 0.0)
        family_vals = [str(v) if v is not None else "" for v in data.get("family", [""] * n)]
        fontface_vals = list(data.get("fontface", [1] * n))
        lineheight_vals = _as_float_array(data.get("lineheight", 1.2), n, 1.2)
        linewidth_vals = _as_float_array(data.get("linewidth", self.label_size), n, self.label_size)

        # Compute per-label bounding boxes and remember their native dimensions.
        boxes = np.zeros((n_valid, 4), dtype=float)
        box_widths = np.zeros(n_valid, dtype=float)
        box_heights = np.zeros(n_valid, dtype=float)
        for i in range(n_valid):
            # Build a textGrob shifted by +label_padding in both axes so its
            # grobWidth/grobHeight incorporate the padding contribution.
            x_anchor = Unit(float(data.iloc[i]["x"]), "native") + self.label_padding
            y_anchor = Unit(float(data.iloc[i]["y"]), "native") + self.label_padding
            t = text_grob(
                label=str(lab[i]),
                x=x_anchor,
                y=y_anchor,
                gp=Gpar(
                    fontsize=float(size_vals[i]) * PT,
                    fontfamily=family_vals[i],
                    fontface=fontface_vals[i],
                    lineheight=float(lineheight_vals[i]),
                ),
                name="text",
            )
            # Rect grob width/height = text width/height + 2*label.padding.
            r_grob = roundrect_grob(
                x=float(data.iloc[i]["x"]),
                y=float(data.iloc[i]["y"]),
                default_units="native",
                width=grid.grob_width(t) + self.label_padding * 2,
                height=grid.grob_height(t) + self.label_padding * 2,
                r=self.label_r,
                gp=Gpar(lwd=float(linewidth_vals[i]) * PT),
                name="box",
            )
            gw = float(convert_width(grid.grob_width(r_grob), "native", valueOnly=True)[0])
            gh = float(convert_height(grid.grob_height(r_grob), "native", valueOnly=True)[0])
            row_x = float(data.iloc[i]["x"])
            row_y = float(data.iloc[i]["y"])
            h = float(hjust_vals[i])
            v = float(vjust_vals[i])
            boxes[i] = [
                row_x - gw * h - box_padding_x + nudge_x_vals[i],
                row_y - gh * v - box_padding_y + nudge_y_vals[i],
                row_x + gw * (1 - h) + box_padding_x + nudge_x_vals[i],
                row_y + gh * (1 - v) + box_padding_y + nudge_y_vals[i],
            ]
            box_widths[i] = gw
            box_heights[i] = gh

        # Resolve seed.
        seed = self.seed
        if seed is None or (isinstance(seed, float) and math.isnan(seed)):
            seed = int(np.random.default_rng().integers(1, 2**31 - 1))
        seed = int(seed)

        p_width = float(convert_width(Unit(1, "npc"), "inch", valueOnly=True)[0])
        p_height = float(convert_height(Unit(1, "npc"), "inch", valueOnly=True)[0])
        p_ratio = p_width / p_height if p_height > 0 else 1.0
        if p_ratio > 1:
            p_ratio = p_ratio ** (1 / (1.15 * p_ratio))

        point_size_series = _as_float_array(data.get("point_size", 1.0), n, 0.0)
        # Every data point is a repulsion source, including unlabeled rows.
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
            # Matches R ``geom-label-repel.R`` (``rlang::inform``).
            from ggrepel_py._options import inform as _inform
            n_skip = int(too_many.sum())
            _inform(
                f"ggrepel: {n_skip} unlabeled data point(s) (too many overlaps). "
                "Consider increasing `max_overlaps`."
            )
        if too_many.all():
            self.set_children(GList())
            return self

        point_size_cm = (
            _as_float_array(data.get("point_size", 1.0), n, 0.0) * PT / _STROKE / 20.0
        )
        point_padding_cm = float(convert_width(to_unit(point_padding), "cm", valueOnly=True)[0])
        width_cm = float(convert_width(Unit(1, "npc"), "cm", valueOnly=True)[0])
        height_cm = float(convert_height(Unit(1, "npc"), "cm", valueOnly=True)[0])

        x_solved = np.asarray(repel["x"], dtype=float)
        y_solved = np.asarray(repel["y"], dtype=float)

        seg_curv = _as_float_array(data.get("segment_curvature", 0.0), n, 0.0)
        seg_ang = _as_float_array(data.get("segment_angle", 90.0), n, 90.0)
        seg_ncp = _as_float_array(data.get("segment_ncp", 1), n, 1).astype(int)
        seg_shape = _as_float_array(data.get("segment_shape", 0.5), n, 0.5)
        seg_square = list(data.get("segment_square", [True] * n))
        seg_sq_shape = _as_float_array(data.get("segment_square_shape", 1.0), n, 1.0)
        seg_inflect = list(data.get("segment_inflect", [False] * n))

        segments: List[Any] = []
        textboxes: List[Any] = []
        for i in range(n_valid):
            if too_many[i]:
                continue
            row = data.iloc[i]
            rect_col = row.get("colour", "black")
            linewidth = float(linewidth_vals[i])
            rect_gp = Gpar(
                col=(None if linewidth == 0 else rect_col),
                fill=_alpha(row.get("fill", "white"), row.get("alpha")),
                lwd=linewidth * PT,
                lty=row.get("linetype", 1),
            )
            text_gp = Gpar(
                col=row.get("colour", "black"),
                fontsize=float(size_vals[i]) * PT,
                fontfamily=family_vals[i],
                fontface=fontface_vals[i],
                lineheight=float(lineheight_vals[i]),
            )
            seg_col = null_default(row.get("segment_colour"), row.get("colour", "black"))
            seg_alpha = null_default(row.get("segment_alpha"), row.get("alpha"))
            arrow_fill = null_default(row.get("arrow_fill"), seg_col)
            segment_gp = Gpar(
                col=_alpha(seg_col, seg_alpha),
                lwd=float(row.get("segment_size", 0.5)) * PT,
                lty=null_default(row.get("segment_linetype"), 1),
                fill=_alpha(arrow_fill, seg_alpha),
            )

            grobs = _make_label_grobs(
                i=i,
                label=str(lab[i]),
                x=Unit(float(x_solved[i]), "native"),
                y=Unit(float(y_solved[i]), "native"),
                x_orig=float(row["x"]),
                y_orig=float(row["y"]),
                box_width=float(box_widths[i]),
                box_height=float(box_heights[i]),
                box_padding=self.box_padding,
                label_padding=self.label_padding,
                label_r=self.label_r,
                point_size=float(point_size_cm[i]),
                point_padding=float(point_padding_cm),
                segment_curvature=float(seg_curv[i]),
                segment_angle=float(seg_ang[i]),
                segment_ncp=int(seg_ncp[i]),
                segment_shape=float(seg_shape[i]),
                segment_square=bool(seg_square[i]),
                segment_square_shape=float(seg_sq_shape[i]),
                segment_inflect=bool(seg_inflect[i]),
                text_gp=text_gp,
                rect_gp=rect_gp,
                segment_gp=segment_gp,
                arrow=self.arrow,
                min_segment_length=self.min_segment_length,
                hjust=float(hjust_vals[i]),
                vjust=float(vjust_vals[i]),
                dim=(width_cm, height_cm),
            )
            if "segment" in grobs:
                segments.append(grobs["segment"])
            textboxes.extend(grobs["textbox"])

        # R order: segments first, then rect+text pairs.
        self.set_children(GList(*(segments + textboxes)))
        return self


# ---------------------------------------------------------------------------
# GeomLabelRepel
# ---------------------------------------------------------------------------


class GeomLabelRepel(GeomLabel):
    """Repulsive label geom. Port of ``GeomLabelRepel``.

    Unlike :class:`GeomTextRepel`, this geom does not support ``rot`` / ``angle``
    (mirroring the R implementation which silently ignores rotation).
    """

    required_aes: Tuple[str, ...] = ("x", "y", "label")
    default_aes: Mapping = Mapping(
        colour="black",
        fill="white",
        size=3.88,
        angle=0,
        alpha=None,
        family="",
        fontface=1,
        lineheight=1.2,
        hjust=0.5,
        vjust=0.5,
        point_size=1,
        linewidth=0.25,
        linetype=1,
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
    )

    def draw_panel(  # type: ignore[override]
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        parse: bool = False,
        na_rm: bool = False,
        box_padding: Any = 0.25,
        label_padding: Any = 0.25,
        point_padding: Any = 1e-6,
        label_r: Any = 0.15,
        label_size: float = 0.25,
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
        if parse:
            # R's geom-label-repel.R:138-140 passes labels through
            # ``parse_safe`` (plotmath).  Python grid has no plotmath
            # renderer, so we reject rather than drop the flag silently.
            raise NotImplementedError(
                "geom_label_repel does not support parse=True "
                "(Python grid has no plotmath renderer)"
            )
        if data is None or len(data) == 0:
            return null_grob()
        # Early-exit when every label is empty — mirrors R geom-label-repel.R:141-143.
        if "label" in data.columns and not any(not_empty(data["label"])):
            return null_grob()
        data = data.reset_index(drop=True).copy()

        # x/y <-> x_orig/y_orig bookkeeping as in R's draw_panel.
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

        def _t(coord: Any, df: pd.DataFrame) -> pd.DataFrame:
            if coord is not None and hasattr(coord, "transform"):
                return coord.transform(df, panel_params)
            return df

        nudges = pd.DataFrame({"x": data["nudge_x"].to_numpy(),
                               "y": data["nudge_y"].to_numpy()})
        nudges = _t(coord, nudges)
        data = _t(coord, data)
        data["nudge_x"] = nudges["x"].to_numpy() - data["x"].to_numpy()
        data["nudge_y"] = nudges["y"].to_numpy() - data["y"].to_numpy()

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
        xlim_inf = np.isinf(xlim_arr)
        ylim_inf = np.isinf(ylim_arr)
        limits_df = pd.DataFrame({"x": xlim_arr, "y": ylim_arr})
        limits_df = _t(coord, limits_df)
        # Restore Inf entries that ``coord_transform`` may have lost
        # (R geom-label-repel.R:191-196).
        if xlim_inf.any():
            limits_df.loc[xlim_inf, "x"] = xlim_arr[xlim_inf]
        if ylim_inf.any():
            limits_df.loc[ylim_inf, "y"] = ylim_arr[ylim_inf]
        limits_df.loc[xlim_na, "x"] = np.array([0.0, 1.0])[xlim_na]
        limits_df.loc[ylim_na, "y"] = np.array([0.0, 1.0])[ylim_na]

        # R's GeomLabelRepel only passes a single coordinate to compute_just
        # (see lines 204/207 — no angle / b argument), so we match that.
        if "vjust" in data.columns and data["vjust"].dtype == object:
            data["vjust"] = compute_just(
                data["vjust"].astype(str).to_numpy(),
                data["y"].to_numpy(),
            )
        if "hjust" in data.columns and data["hjust"].dtype == object:
            data["hjust"] = compute_just(
                data["hjust"].astype(str).to_numpy(),
                data["x"].to_numpy(),
            )

        tree = LabelRepelTree(
            data=data,
            lab=data["label"].tolist(),
            limits=limits_df,
            box_padding=to_unit(box_padding),
            label_padding=to_unit(label_padding),
            point_padding=to_unit(point_padding),
            label_r=to_unit(label_r),
            label_size=label_size,
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
        return ggname("geom_label_repel", tree)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def geom_label_repel(
    mapping: Any = None,
    data: Any = None,
    stat: str = "identity",
    position: Any = "identity",
    parse: bool = False,
    *,
    box_padding: Any = 0.25,
    label_padding: Any = 0.25,
    point_padding: Any = 1e-6,
    label_r: Any = 0.15,
    label_size: float = 0.25,
    min_segment_length: Any = 0.5,
    arrow: Any = None,
    force: float = 1.0,
    force_pull: float = 1.0,
    max_time: float = 0.5,
    max_iter: int = 10000,
    max_overlaps: Any = _UNSET,
    nudge_x: Any = 0,
    nudge_y: Any = 0,
    xlim: Any = (None, None),
    ylim: Any = (None, None),
    na_rm: bool = False,
    show_legend: Any = None,
    direction: str = "both",
    seed: Any = None,
    verbose: Any = _UNSET,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Repulsive label layer. Port of R ``geom_label_repel()``."""
    if direction not in ("both", "x", "y"):
        # Matches R's ``match.arg(direction)`` validation.
        raise ValueError(
            f"`direction` must be one of 'both', 'x', 'y'; got {direction!r}"
        )
    # Resolve option-backed defaults at call time (geom-label-repel.R:25,34).
    if max_overlaps is _UNSET:
        max_overlaps = _get_option("ggrepel.max.overlaps", 10)
    if verbose is _UNSET:
        verbose = _get_option("verbose", False)
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
        "label_padding": to_unit(label_padding),
        "point_padding": to_unit(point_padding),
        "label_r": to_unit(label_r),
        "label_size": label_size,
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
        geom=GeomLabelRepel,
        stat=stat,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params=params,
    )
