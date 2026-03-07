from __future__ import annotations

import os
import tempfile
from pathlib import Path


def configure_runtime() -> None:
    if "MPLCONFIGDIR" not in os.environ:
        mpl_default = Path.home() / ".matplotlib"
        if not mpl_default.exists() or not os.access(mpl_default, os.W_OK):
            os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="matplotlib-")
    if "XDG_CACHE_HOME" not in os.environ:
        xdg_cache = Path.home() / ".cache"
        if not xdg_cache.exists() or not os.access(xdg_cache, os.W_OK):
            os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp(prefix="xdg-cache-")
