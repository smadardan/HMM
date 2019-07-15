
"""
utils.utilities
~~~~~

functions that are general and can help a lot of different applications
"""

import logging
import json


import config.config_vars as cnf
from src.HMM import Hmm


def configure_logging():
    """
    initialize the logging process
    :return:
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # create a file handler
    handler = logging.FileHandler(cnf.LOG_FILE_NAME)
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter('%(asctime)s: %(name)s: %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    # print to stdout
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)
    logger.addHandler(consoleHandler)
    return logger


def set_config(logger):
    """
    uploads the observation and hidden states data
    :param logger: log writer
    :return: dict, the data, parsed
    """
    with open(cnf.INPUT_FILE_NAME) as f:
        data = json.load(f)
    logger.info('observations and hidden layers parametersL {}'.format(data))
    return data

def generate_data(model, logger):
    """
    generates k different observations and also shows what is their probability with specific hidden states
    :param model:
    :param logger:
    :return:
    """
    hmm = Hmm(model['a'], model['b'], model['pi'], model['obs_names'], model['hidden_names'], logger)
    obs, hidden, p = hmm.generate(cnf.GENERATE_COUNT)
    return obs, hidden, p

def print_compare(obs, hidden, p, v_output, b_output, v_prob, b_prob, logger):
    """
    prints in logs all of the results
    :param obs: list of strings, the generated observations
    :param hidden: list of strings, sampled hidden states for each day
    :param p: list of floats, the calculates probabilities for each samples hidden state
    :param v_output: list of strings: viterbi hidden states
    :param b_output: list of string, backward-forward hidden states
    :param v_prob: list of floats, probabilities for viterbi
    :param b_prob: list of floats, probabilities for backward-forward
    :param logger: object, write into log
    :return:
    """
    logger.info('for observation: {}'.format(obs))
    logger.info('sampling hidden: {}'.format(hidden))
    logger.debug('probability: {}'.format(p))
    logger.info('viterbi hidden: {}'.format(v_output))
    logger.debug('viterbi probability: {}'.format(v_prob))
    logger.info('backward-forward hidden: {}'.format(b_output))
    logger.debug('backward-forward probability: {}'.format(b_prob))
    return
