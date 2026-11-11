"""The directory that holds the package's configuration.

The run descriptions, the resonance catalog, and the other configuration
files live in ``config/``.  In a checkout, ``config/`` sits next to the
package, in ``dev/mixing2006``.  An installed wheel carries a copy of it
inside the package, in ``na49_mixing/_data/config``, which ``setup.py``
places there at build time.  ``nested_root()`` returns the directory that
holds ``config/`` in either case.  A checkout always takes precedence, so
the paths of a checkout are unchanged.
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["PACKAGE_DIR", "BUNDLED_ROOT", "nested_root"]

PACKAGE_DIR = Path(__file__).resolve().parent
#: the copy of ``config/`` that an installed wheel carries
BUNDLED_ROOT = PACKAGE_DIR / "_data"


def nested_root() -> Path:
    """The directory that holds ``config/``.

    This is the checkout's ``dev/mixing2006`` when ``config/`` sits next to
    the package, and the bundled copy when only that exists."""
    checkout = PACKAGE_DIR.parent
    if (checkout / "config").is_dir() or not (BUNDLED_ROOT / "config").is_dir():
        return checkout
    return BUNDLED_ROOT
