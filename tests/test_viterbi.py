"""
tests.viterbi
~~~~~~~~~~~~~~~~
here we will write all of the unit test for the viterbi class
"""

from src.viterbi import Viterbi
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
inst = Viterbi(hmm_vars['a'], hmm_vars['b'], hmm_vars['pi'], hmm_vars['obs_names'],
               hmm_vars['hidden_names'], ['Clean'], logger)


def test_calc_first_observation():
    out = inst.calc_first_observation([0, 1])
    assert out == {0: [0.45, 0.1]}


def test_calc_probas_for_state():
    out = inst.calc_probas_for_state(0, {0: [0.5, 0.1]}, 1, 0)
    assert out == [0.315, 0.027]


def test_update_variables():
    remember, earlier = inst.update_variables([0.1, 0.2], {0: [0.1, 0.2], 1: [0.1]}, {0: [1], 1: []}, 1)
    assert remember == {0: [0.1, 0.2], 1: [0.1, 0.2]}
    assert earlier == {0: [1], 1: [1]}


def test_keep_best_hidden_for_day():
    out = inst.keep_best_hidden_for_day({0: [0.1, 0.2], 1: [0.1]}, {0: [1], 1: [1]}, [0, 0])
    assert out == [1, 0]


def test_viterbi():
    out, p = inst.viterbi()
    assert out == ['Rainy']
    assert p == {0: [0.05, 0.4]}
