"""Characterisation tests for the pure helpers the scoring core is built from.

They pin what the shipped code returns today, so a later phase that moves these
functions cannot change their behaviour unnoticed.
"""

import numpy as np
import pandas as pd
import pytest

from fpsearch import _once, normalise, slot_merge
from metric import mrr, reciprocal_rank, validate
from shifted import analog_term


def test_slot_merge():
    assert slot_merge(list("ABC"), list("XY"), (2, 4)) == list("AXBYC")
    assert slot_merge(list("ABC"), [], (2, 4)) == list("ABC")
    assert slot_merge(list("A"), list("XYZ"), (4,)) == list("AXYZ")
    assert slot_merge([], list("XY"), (2, 4, 6, 8, 10, 24)) == list("XY")
    assert slot_merge(list("ABCDEFGHIJ"), list("XYZ"), (2, 4, 6), n=5) == list("AXBYC")


def test_once_calls_f_once():
    calls = []

    def f():
        calls.append(1)
        return object()

    g = _once(f)
    first = g()
    second = g()
    assert len(calls) == 1
    assert first is second


def test_normalise():
    t = np.array([1.0, 2.0, 3.0])
    assert normalise(t, "") is t
    assert np.allclose(normalise(t, "rank"), [0, 0.5, 1])
    assert np.allclose(normalise(np.array([5.0]), "z"), [0.0])


def test_analog_term():
    assert np.allclose(analog_term([[255]], [[255], [15]], [0.5, 1.0], p=3), [0.5])
    assert np.allclose(analog_term([[255]], [], [], p=3), [0.0])


def test_reciprocal_rank():
    # Same molecule, different SMILES: the InChI-style key matches.
    assert reciprocal_rank(["C", "OCC"], "CCO") == 0.5
    # Rank 26 is past K = 25, so it does not count.
    assert reciprocal_rank(["C"] * 25 + ["CCO"], "CCO") == 0.0
    assert mrr({"a": ["CCO"]}, {"a": "CCO", "b": "C"}) == 0.5


def test_validate_rejects():
    too_many = pd.DataFrame({"molecule_id": ["a"], "smiles": [";".join(["C"] * 26)]})
    with pytest.raises(AssertionError):
        validate(too_many)

    duplicated = pd.DataFrame({"molecule_id": ["a", "a"], "smiles": ["C", "CC"]})
    with pytest.raises(AssertionError):
        validate(duplicated)
