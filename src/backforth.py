"""
src.backward-forward
~~~~~~~~~~~~

computes the posterior marginals of all hidden state variables given a sequence of observations/emissions
"""

import numpy as np
from src.HMM import Hmm


class BackwardForward(Hmm):
    def __init__(self, a, b, pi, obs_names, hidden_names, cs, logger):
        Hmm.__init__(self, a, b, pi, obs_names, hidden_names, logger)
        self.cs = cs

    def backward_forward(self):
        """
        calls funcs backward and forward to fins alphas and betas.
        calls smoothing to calc the probability based on alphas and betas
        from probabilities calls funcs find_max_proba and convert_hidden_num_to_name to return the
        hidden state
        :return:
        """
        obs = self.convert_obs_names_to_nums(self.cs)
        mat = self.obs_to_trans(obs)
        alphas = self.forward(mat)
        betas = self.backward(mat)
        probas = self.smoothing(alphas, betas)
        hidden = self.find_max_probas(probas)
        hidden_names = self.convert_hidden_num_to_name(hidden)
        self.logger.debug('backward-forward solution: {}'.format(hidden_names))
        return hidden_names, probas

    def forward(self, mat):
        """
        gets the observation matrix and do matrix calculations to
        find the alphas.
        normalize each alpha and return a list of alphas.
        :param mat: np.array - observation matrix
        :return: list of forward probabilities
        """
        norm_alphas = []
        for count, obs in enumerate(mat):
            if count == 0:
                alpha = obs @ np.array(self.a) @ np.array(self.pi)
                normalize_factor = 1 / np.sum(alpha, axis=0)
                norm_alphas.append((normalize_factor * alpha))
            else:
                alpha = obs @ np.array(self.a) @ norm_alphas[count - 1]
                normalize_factor = 1 / np.sum(alpha, axis=0)
                norm_alphas.append((normalize_factor * alpha))
        return [np.array(self.pi)] + norm_alphas

    def backward(self, mat):
        """
        gets the observation matrix and do matrix calculations to
        find the betas.
        normalize each alpha and return a list of betas
        adds the final condition (1, 1) and sends everything backwards
        :param mat: np.array - observation matrix
        :return: backward probabilities
        """
        norm_betas = []
        for count, obs in enumerate(mat[::-1]):
            if count == 0:
                beta = np.array(self.a) @ obs @ np.array([1] * len(self.hidden_names))
                normalize_factor = 1 / np.sum(beta, axis=0)
                norm_betas.append((normalize_factor * beta))
            else:
                beta = np.array(self.a) @ obs @ norm_betas[count - 1]
                normalize_factor = 1 / np.sum(beta, axis=0)
                norm_betas.append((normalize_factor * beta))
        norm_betas = [np.array([1] * len(self.hidden_names))] + norm_betas
        return norm_betas[::-1]

    def smoothing(self, alphas, betas):
        """
        takes alphas and betas, multiply them and normalize.
        return the outcome which is the probability
        :param alphas: forward probabilities
        :param betas: backward probabilities
        :return: list of final probabilities after normalization
        """
        probas = []
        for count, alpha, beta in (zip(range(len(alphas)), alphas, betas)):
            multi = alpha * beta
            normalize_factor = 1 / np.sum(multi, axis=0)
            probas.append((normalize_factor * multi))
        self.logger.info('Probabilities, backward forward\n{}'.format(probas))
        return probas

    def find_max_probas(self, probas):
        """
        finds the index with the max value and append the index to a new list
        the indexes are the hidden state
        :param probas: list of probabilities after normalization
        :return: the max hidden state which is the chosen hidden state
        """
        hidden_state = []
        for prob in probas:
            hidden_state.append(np.argmax(prob))
        self.logger.debug('maximum probability for each hidden state: {}'.format(hidden_state))
        return hidden_state

    def obs_to_trans(self, obs):
        """
        :param obs: observation string
        :return: list of matrices where the only probabilities left are for the observation state
        that is known to happen
        """
        mat_obs = []
        for ob in obs:
            mat_obs.append(np.diag([self.b[i][ob] for i in range(len(self.hidden_names))]))
        return mat_obs
