#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
main
~~~~~

in this project we will use 2 hidden markov models - viterbi and backward-forward to see probabilities for
problems with transitioning states and hidden layers.
"""

import pytest

import utils.utilities as utl
from src.viterbi import Viterbi
from src.backforth import BackwardForward
import config.config_vars as cnf

def main():
    """
    starts logging and insert data to population to initiate the genetic algorithm
    :param population_size: int, how many items are allowed in the knapsack
    :param max_weight: int, what is the maximum weight allowed in the knapsack
    :param num_iterations: how many iterations can the genetic algorithm perform
    :return:
    """
    pytest.main()
    logger = utl.configure_logging()
    logger.info('## Started ##')

    try:
        model = utl.set_config(logger)
    except IOError:
        logger.error('Failed to open file', exc_info=True)
        return
    for i in range(cnf.ITERATION):
        logger.info('\n{}.'.format(i))
        obs, hidden, p = utl.generate_data(model, logger)
        v_inst = Viterbi(model['a'], model['b'], model['pi'], model['obs_names'], model['hidden_names'], obs, logger)
        b_inst = BackwardForward(model['a'], model['b'], model['pi'], model['obs_names'],
                                 model['hidden_names'], obs, logger)
        v_output, v_prob = v_inst.viterbi()
        b_output, b_prob = b_inst.backward_forward()
        utl.print_compare(obs, hidden, p, v_output, b_output, v_prob, b_prob, logger)
    logger.info('## Finished ##')
    return


if __name__ == "__main__":
    main()
