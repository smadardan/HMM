"""
tests.backforth
~~~~~~~~~~~~~~~~
here we will write all of the unit test for the backforth class
"""
import numpy as np

from src.backforth import BackwardForward
from tests.test_HMM import LoggerMock


# initialize for all tests
hmm_vars = {
    "pi": [0.5, 0.5],
    "a": [[0.7, 0.3], [0.3, 0.7]],
    "b": [[0.9, 0.1], [0.2, 0.8]],
    "obs_names": ["Dirty", "Clean"],
    "hidden_names": ["Sunny", "Rainy"]
}
logger = LoggerMock()
inst = BackwardForward(hmm_vars['a'], hmm_vars['b'], hmm_vars['pi'], hmm_vars['obs_names'], hmm_vars['hidden_names'],
                       ['Clean'], logger)


def test_forward():
    out = inst.forward(np.array([[0, 1]]))
    assert np.array_equal(out[0], np.array([0.5, 0.5]))
    assert 1.0 == out[1]


def test_backward():
    out = inst.backward(np.array([[0, 1]]))
    assert 1.0 == out[0]
    assert np.array_equal(out[1], np.array([1, 1]))


def test_smoothing():
    out = inst.smoothing([np.array([[0.1, 0.2]]), np.array([[0.1, 0.2]])], [0.1, 0.1])
    assert np.array_equal(out[0], np.array([[1, 1]]))
    assert np.array_equal(out[1], np.array([[1, 1]]))


def test_find_max_probas():
    out = inst.find_max_probas([[0.1, 0.2], [0.2, 0.1], [0.3, 0.2]])
    assert out == [1, 0, 0]


def test_obs_to_trans():
    out = inst.obs_to_trans([0])
    assert np.array_equal(out[0], np.array([[0.9, 0], [0, 0.2]]))
