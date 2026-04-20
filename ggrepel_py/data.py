"""Bundled datasets for ggrepel_py tutorials and tests."""

from __future__ import annotations

from importlib.resources import files

import pandas as pd

__all__ = ["load_mtcars"]


def load_mtcars() -> pd.DataFrame:
    """Return the `mtcars` dataset as a `pandas.DataFrame`.

    32 rows × 12 columns. The first column ``car`` holds the row names from
    base R's ``datasets::mtcars``; the remaining 11 numeric columns match base
    R's column order: mpg, cyl, disp, hp, drat, wt, qsec, vs, am, gear, carb.
    """
    path = files("ggrepel_py.resources") / "mtcars.csv"
    with path.open("r", encoding="utf-8") as fh:
        return pd.read_csv(fh)
