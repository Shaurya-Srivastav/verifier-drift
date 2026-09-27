#!/usr/bin/env python3
"""Dependency-free test runner (works without pytest).

Usage:  python scripts/run_tests.py
"""
from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

TEST_MODULES = ["test_forms", "test_verifier", "test_audit", "test_pilot_smoke"]


def main() -> int:
    passed = failed = 0
    for modname in TEST_MODULES:
        mod = importlib.import_module(modname)
        for name in dir(mod):
            if name.startswith("test_"):
                fn = getattr(mod, name)
                try:
                    fn()
                    passed += 1
                    print(f"PASS {modname}.{name}")
                except Exception:
                    failed += 1
                    print(f"FAIL {modname}.{name}")
                    traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
