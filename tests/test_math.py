import math

import pytest
import pandas as pd

from decision_trees import metrics

@pytest.fixture
def df():
    yield pd.read_csv('tests/res/data.csv')

@pytest.mark.parametrize("sfty", [("low", 0.0), ("med", 1.0), ("high", 0.918)])
def test_entropy(df: pd.DataFrame, sfty: (str, float)):
    type, value = sfty
    entropy = metrics.entropy(df[df["safety"] == type], "decision")
    assert math.isclose(entropy, value, rel_tol=1e-3)

def test_cond_entropy(df: pd.DataFrame):
    cond_entropy = metrics.cond_entropy(df, "safety", "decision")
    assert math.isclose(cond_entropy, 0.7508, rel_tol=1e-3)

def test_gain(df: pd.DataFrame):
    gain = metrics.gain(df, "safety", "decision")
    assert math.isclose(gain, 0.2492, rel_tol=1e-3)

def test_intrinsic_info(df: pd.DataFrame):
    intrinsic_info = metrics.intrinsic_info(df, "safety")
    assert math.isclose(intrinsic_info, 1.371, rel_tol=1e-4)

def test_gain_ratio(df: pd.DataFrame):
    gain_ratio = metrics.gain_ratio(df, "safety", "decision")
    assert math.isclose(gain_ratio, 0.1818, rel_tol=1e-3)
