"""Options system + diagnostic messaging.

Port of R's ``getOption`` / ``options`` used in the ggrepel factory
functions, plus ``rlang::inform`` used by the solver's ``verbose`` path.

R options read by ggrepel (see ``R/geom-text-repel.R``,
``R/geom-label-repel.R``):

- ``"ggrepel.max.overlaps"`` — default ``10``.
- ``"verbose"`` — default ``FALSE``.  In R this is the global ``verbose``
  option shared with ``base R`` / ``ggplot2`` etc.  Because Python has
  no equivalent global option namespace, ggrepel_py reads from its own
  options dict; set ``options(ggrepel_py.set_option("verbose", True))``
  to enable messages from all ``geom_*_repel`` calls.

Public API
----------
``get_option(name, default=None)``
    Return the current value, else *default*.  Mirrors R's
    ``getOption(name, default=...)``.

``set_option(name, value)``
    Set an option and return the previous value.  Mirrors R's
    ``options(name = value)``.

``inform(msg)``
    Emit a diagnostic message, mirroring R's ``rlang::inform``.  Uses a
    dedicated Python logger (``"ggrepel_py"``) at INFO level so users can
    silence or redirect via standard :mod:`logging` configuration.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict

__all__ = ["get_option", "set_option", "inform"]


_OPTIONS: Dict[str, Any] = {}


def get_option(name: str, default: Any = None) -> Any:
    """Return the value of option *name*, or *default* if unset.

    Mirrors R's ``getOption(name, default = ...)``.
    """
    return _OPTIONS.get(name, default)


def set_option(name: str, value: Any) -> Any:
    """Set option *name* to *value*; return the previous value.

    Mirrors R's ``options(name = value)`` which returns the old value.
    ``value=None`` removes the option entirely (matching R's
    ``options(name = NULL)`` which unsets the option so subsequent
    ``getOption(name, default = X)`` returns the default).
    """
    prev = _OPTIONS.get(name)
    if value is None:
        _OPTIONS.pop(name, None)
    else:
        _OPTIONS[name] = value
    return prev


# Dedicated logger.  R's ``rlang::inform`` writes directly to stderr
# without going through the warning system (so it isn't intercepted by
# ``-W error`` / ``pytest.warns``).  We achieve the same by routing
# through a named logger that ships a default stderr handler at INFO
# level; applications can override either by configuring the logger.
_logger = logging.getLogger("ggrepel_py")
if not any(
    isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stderr
    for h in _logger.handlers
):
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)
if _logger.level == logging.NOTSET:
    _logger.setLevel(logging.INFO)
# Do not propagate to the root logger — R's ``inform`` isn't re-broadcast.
_logger.propagate = False


def inform(msg: str) -> None:
    """Emit a diagnostic message, like R's ``rlang::inform``.

    The message is logged at ``INFO`` level on the ``"ggrepel_py"``
    logger (stderr by default).  Silence with:

    >>> import logging
    >>> logging.getLogger("ggrepel_py").setLevel(logging.WARNING)
    """
    _logger.info(str(msg))
