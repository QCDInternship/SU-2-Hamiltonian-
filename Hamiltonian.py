"""Compatibility entry point for the historical two-spin demonstration.

All Hamiltonian construction and analysis live in :mod:`ising_limit_model`.
New code should import that module directly or run ``python ising_limit_model.py``.
"""

from __future__ import annotations

import sys

from ising_limit_model import main


if __name__ == "__main__":
    raise SystemExit(main([*sys.argv[1:], "--legacy-two-site-output"]))
