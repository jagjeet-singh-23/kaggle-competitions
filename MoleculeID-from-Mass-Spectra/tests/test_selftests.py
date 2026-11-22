"""Run each module's own data-free selftest under pytest."""

import features
import formula
import library
import shifted


def test_formula_selftest():
    formula.selftest()


def test_shifted_selftest():
    shifted.selftest()


def test_library_selftest():
    library.selftest()


def test_features_selftest():
    features.selftest()
