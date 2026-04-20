# ggrepel_py

Python port of the R [**ggrepel**](https://ggrepel.slowkow.com/) package
(version 0.9.8.9999) — non-overlapping text and label layout for
`ggplot2_py`.

## Features

- `geom_text_repel()` — repelling text labels
- `geom_label_repel()` — repelling labels with a rounded rectangle background
- `position_nudge_repel()` — preserve original data-point coordinates after nudging
- Reproducible layouts via `seed=`
- Curved segments, arrow heads, snake-case aesthetic names (e.g. `segment_colour`, `bg_colour`)

## Installation

```bash
pip install -e .
```

Requires `ggplot2_py`, `grid_py`, `scales_py`, `gtable_py`, and `numpy`.
The C++ force-directed solver is built via pybind11 at install time.

## Quick start

```python
import pandas as pd
from ggplot2_py import ggplot, aes, geom_point
from ggrepel_py import geom_text_repel
from ggrepel_py.data import load_mtcars

dat = load_mtcars()
dat = dat[(dat['wt'] > 2.75) & (dat['wt'] < 3.45)]

(ggplot(dat, aes(x='wt', y='mpg', label='car'))
 + geom_point(color='red')
 + geom_text_repel(seed=42))
```

## Learn more

- [Getting started](tutorials/ggrepel.ipynb) — basic usage, key options
- [Examples (V2)](tutorials/examples_v2.ipynb) — hidden labels, nudging, curves, arrows
- [API reference](api.md)

## Relationship to R ggrepel

This port preserves the semantics of the force-directed solver (`repel_boxes2`)
exactly — quantitative parity against R reference fixtures is ≥ 0.9999999
Pearson correlation for non-coincident points.

Intentional deviations from R ggrepel:

- **Snake-case API**: dotted R aesthetic names (`segment.colour`, `bg.colour`,
  `point.size`, `min.segment.length`) become snake_case in Python.
- **RNG stream**: the C++ solver uses `std::mt19937` seeded from the integer
  `seed=` argument. For non-coincident points this produces identical
  converged positions to R within numerical tolerance; for coincident points,
  per-label identity is scrambled but the overall label spread is preserved.
- **`element_text_repel` theme element**: not ported (out of scope).
- **`marquee`-formatted labels**: not ported (out of scope).
