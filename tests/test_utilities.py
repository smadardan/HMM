"""
utils.utilities
~~~~~~~~~~~~~~~~
here we will write all of the unit test for the utility functions class
"""
from unittest.mock import patch

import utils.utilities as utl
from tests.test_HMM import LoggerMock
import config.config_vars as cnf

# initialize for all tests
hmm_vars = {
    "pi": [0.5, 0.5],
    "a": [[0.7, 0.3], [0.3, 0.7]],
    "b": [[0.9, 0.1], [0.2, 0.8]],
    "obs_names": ["Dirty", "Clean"],
    "hidden_names": ["Sunny", "Rainy"]
}
logger = LoggerMock()


@patch('logging.getLogger')
def test_configure_logging(getLogger):
    utl.configure_logging()
    assert getLogger.called


@patch('json.load')
def test_set_config(load):
    utl.set_config(logger)
    assert load.called


def test_generate_data():
    obs, hidden, p = utl.generate_data(hmm_vars, logger)
    assert len(obs) == cnf.GENERATE_COUNT
    assert len(hidden) == cnf.GENERATE_COUNT
    assert 'Clean' in obs or 'Dirty' in obs
    assert 'Sunny' in hidden or 'Rainy' in hidden
    assert 1 > p > 0
