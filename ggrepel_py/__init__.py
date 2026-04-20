"""ggrepel_py — non-overlapping text/label layout for ggplot2_py.

Python port of the R package ggrepel (v0.9.8.9999).
"""

from ggrepel_py.geom_text_repel import (
    GeomTextRepel,
    TextRepelTree,
    geom_text_repel,
)
from ggrepel_py.geom_label_repel import (
    GeomLabelRepel,
    LabelRepelTree,
    geom_label_repel,
)
from ggrepel_py.position_nudge_repel import (
    PositionNudgeRepel,
    position_nudge_repel,
)

__version__ = "0.9.8.9999"

__all__ = [
    "__version__",
    "GeomTextRepel",
    "TextRepelTree",
    "geom_text_repel",
    "GeomLabelRepel",
    "LabelRepelTree",
    "geom_label_repel",
    "PositionNudgeRepel",
    "position_nudge_repel",
]
