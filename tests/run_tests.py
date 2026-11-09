# -*- coding: utf-8 -*-
"""추가 패키지 없이 배포 라이브러리 검사: python tests/run_tests.py"""
from pathlib import Path
import inspect
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_env
import test_regressions

suite = unittest.TestSuite()
for name, fn in inspect.getmembers(test_env, inspect.isfunction):
    if name.startswith('test_'):
        suite.addTest(unittest.FunctionTestCase(fn))
suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(test_regressions))
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
