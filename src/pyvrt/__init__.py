from __future__ import annotations

from .config import DEFAULT_CONFIG, build_config

__all__ = ["DEFAULT_CONFIG", "build_config", "run"]
__version__ = "0.1.0"


def run(config):
    from .pipeline import run as _run

    return _run(config)
