"""
src.HMM
~~~~~~~~~
in this file you can find the utilities for viterbi and backward forward. this will help to take real example and
convert to hidden markov model input
"""

import random


class Hmm(object):
    def __init__(self, a, b, pi, obs_names, hidden_names, logger):
        self.a = a
        self.b = b
        self.pi = pi
        self.obs_names = obs_names
        self.hidden_names = hidden_names
        self.logger = logger

    def _sample_category(self, ps):
        p = random.random()
        s = ps[0]
        i = 0
        while s < p and i < len(ps):
            i += 1
            s += ps[i]
        self.logger.debug('the samples: {}'.format(i))
        return i

    def convert_obs_names_to_nums(self, xs):
        """
        convert the observation names to numbers so we could append by location
        :param xs:
        :return: the observations as numbers - 0 for obs_names in the 0 place and
        1 for the name in the 1st place
        """
        obs_nums = [self.obs_names.index(i) for i in xs]
        return obs_nums

    def convert_hidden_num_to_name(self, hidden):
        """
        takes a string of 1 and 0, and returns the name of the hidden
        state
        :param hidden: list of ints (1 or 0)
        :return: list of hidden names
        """
        hidden_outcome = []
        for val in hidden:
            hidden_outcome.append(self.hidden_names[val])
        return hidden_outcome

    def generate(self, size):
        zs, xs = [], []
        seq_prob = 1.
        z_t = None
        for i in range(size):
            if i == 0:
                z_t = self._sample_category(self.pi)
                seq_prob *= self.pi[z_t]
            else:
                a_z_t = self.a[z_t]
                z_t = self._sample_category(a_z_t)
                seq_prob *= a_z_t[z_t]
            x_t = self._sample_category(self.b[z_t])
            zs.append(self.hidden_names[z_t])
            xs.append(self.obs_names[x_t])
            seq_prob *= self.b[z_t][x_t]
        self.logger.debug('the generated data: \nxs: {}\nzs: {}\nseq_prob: {}'.format(xs, zs, seq_prob))
        return xs, zs, seq_prob
