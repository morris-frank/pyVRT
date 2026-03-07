#!/usr/bin/env python3
from __future__ import annotations

from pyvrt import DEFAULT_CONFIG, build_config, run
from pyvrt.cli import main, make_arg_parser

__all__ = ["DEFAULT_CONFIG", "build_config", "run", "make_arg_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
