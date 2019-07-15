"""
tests.HMM
~~~~~~~~~~~~~~~~
here we will write all of the unit test for the HMM class
"""

from src.HMM import Hmm


class LoggerMock:
    def __init__(self):
        return

    def debug(self, text):
        pass

    def info(self, test):
        pass


# initialize for all tests
hmm_vars = {
    "pi": [0.5, 0.5],
    "a": [[0.7, 0.3], [0.3, 0.7]],
    "b": [[0.9, 0.1], [0.2, 0.8]],
    "obs_names": ["Dirty", "Clean"],
    "hidden_names": ["Sunny", "Rainy"]
}
logger = LoggerMock()
inst = Hmm(hmm_vars['a'], hmm_vars['b'], hmm_vars['pi'], hmm_vars['obs_names'], hmm_vars['hidden_names'], logger)


def test_sample_category():
    output = inst._sample_category(inst.pi)
    assert output == 0 or output == 1


def test_convert_obs_names_to_nums():
    obs = ['Dirty', 'Dirty', 'Clean']
    output = inst.convert_obs_names_to_nums(obs)
    assert output == [0, 0, 1]


def test_convert_hidden_num_to_name():
    input_list = [0, 1, 1]
    output = inst.convert_hidden_num_to_name(input_list)
    assert output == ['Sunny', 'Rainy', 'Rainy']


def test_generate():
    xs, zs, seq_prob = inst.generate(1)
    assert xs == ['Clean'] or xs == ['Dirty']
    assert zs == ['Sunny'] or zs == ['Rainy']
    assert 1 > seq_prob > 0
