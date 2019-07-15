"""
src.viterbi
~~~~~~~~~~~~

dynamic programming algorithm for finding the most likely sequence of hidden states called the Viterbi path
that results in a sequence of observed events
"""

from collections import defaultdict

from src.HMM import Hmm


class Viterbi(Hmm):
    def __init__(self, a, b, pi, obs_names, hidden_names, xs, logger):
        Hmm.__init__(self, a, b, pi, obs_names, hidden_names, logger)
        self.xs = xs

    def viterbi(self):
        """
        calculate the probability for each day to be A or B hidden state given the last day was A or B.
        saves the best probability of the day and the state that led to it.
        :return: list. the probability for each day. this road was chosen as the most probable
        """
        earlier = defaultdict(list)
        observations = self.convert_obs_names_to_nums(self.xs)
        remembering = self.calc_first_observation(observations)

        earlier[0] = [None] * len(self.hidden_names)

        for day, obs in zip(range(1, len(self.xs)), observations[1:]):  # for each day and observation
            for state in range(len(self.hidden_names)):  # for each hidden option
                probas_for_state = self.calc_probas_for_state(state, remembering, day, obs)
                remembering, earlier = self.update_variables(probas_for_state, remembering, earlier, day)

        hidden = self.keep_best_hidden_for_day(remembering, earlier, self.xs)
        output = self.convert_hidden_num_to_name(hidden)
        self.logger.debug('viterbi solution: {}'.format(output))
        return output, remembering

    def calc_first_observation(self, observations):
        """
        using pi to calculate the first observations for each hidden states
        :param observations: list ints, observations
        :return: remembering: list of initial probability for each hidden state
        """
        remembering = defaultdict(list)
        for i in range(len(self.hidden_names)):
            remembering[0].append(self.pi[i] * self.b[i][observations[0]])
        return remembering

    def calc_probas_for_state(self, state, remembering, day, obs):
        """
        calculate for each day and each hidden state the probability by using the day before
        :param state: int, the hidden state that
        :param remembering: dict of previous probabilities
        :param day: int, the day we want to calculate
        :param obs: int, the present observation
        :return: probas_for_state, list of probabilities for each hidden state
        """
        probas_for_state = []
        for other_state in range(len(self.hidden_names)):
            if other_state == state:
                probas_for_state.append(remembering[day - 1][state] * self.a[state][state] * self.b[state][obs])
            else:
                probas_for_state.append(
                    remembering[day - 1][other_state] * self.a[other_state][state] * self.b[state][obs])
        return probas_for_state

    def update_variables(self, probas_for_state, remembering, earlier, day):
        max_value = max(probas_for_state)
        max_index = probas_for_state.index(max_value)
        remembering[day].append(max_value)
        earlier[day].append(max_index)
        self.logger.debug('max value: {} for day: {}'.format(max_index, day))
        return remembering, earlier

    def keep_best_hidden_for_day(self, remembering, earlier, xs):
        """
        finds best probability for the last day, take its index and run on the days that led to this day in reverse
        :param remembering: dict, best probabilities for hidden states
        :param earlier: dict, the days that led to this day
        :param xs:
        :return: the
        """
        hidden = []
        last_day = len(xs) - 1
        max_value = max(remembering[last_day])
        max_index = remembering[last_day].index(max_value)
        hidden.append(max_index)
        days = last_day
        self.logger.debug('last day best probability: {}. all days probabilities: {}'.format(max_value, remembering))

        while days:
            the_day_before_was = earlier[days][max_index]
            hidden.append(the_day_before_was)
            days -= 1
            max_index = the_day_before_was
        return hidden[::-1]
