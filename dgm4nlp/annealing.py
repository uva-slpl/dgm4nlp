"""
:Authors: - Wilker Aziz
"""
from collections import defaultdict, deque
import numpy as np


def generalised_logistic(x, lower_asymp=0., upper_asymp=1., growth_rate=0.001, starting_time=5000, nu=1., q=1., c=1.):
    """
    https://en.wikipedia.org/wiki/Generalised_logistic_function

    :param x:
    :param lower_asymp: A
    :param upper_asymp: K
    :param growth_rate: B
    :param starting_time: M
    :param nu: \nu > 0
    :param q: Q
    :param c: C
    :return: A + (K - A) / (C + Q * exp(- B * (x - M)))^(1. / \nu)
    """
    return lower_asymp + (upper_asymp - lower_asymp) / np.power(c + q * np.exp(- growth_rate * (x - starting_time)), 1. / nu)


class MovingAverage:

    def __init__(self, max_size=None):
        self._data = defaultdict(deque)
        self._max_size = max_size

    def register(self, name, value):
        obs = self._data[name]
        obs.append(value)
        if self._max_size and len(obs) > self._max_size:
            obs.popleft()

    def register_all(self, pairs):
        for name, value in pairs:
            self.register(name, value)

    def mean(self, name):
        obs = self._data[name]
        return np.mean(obs, dtype=float) if obs else 0.

    def sum(self, name):
        obs = self._data[name]
        return np.sum(obs, dtype=float) if obs else 0.

    def tostring(self, keys):
        return ' '.join('{}={:.4f}'.format(key, self.mean(key)) for key in keys)

    def __str__(self):
        return ' '.join('{}={:.4f}'.format(name, self.mean(name)) for name in sorted(self._data.keys()))


class Schedule:

    def __init__(self, initial=1., final=1., step=0., wait=0, nb_updates=1, wait_value=None, name='alpha'):
        self._initial = initial
        self._final = final
        self._step = step
        self._wait = wait
        self._nb_updates = nb_updates
        self._alpha = initial
        self._alpha_while_waiting = wait_value if wait_value is not None else initial
        self._counter = 0
        self._t = 0
        self.name = name

    def alpha(self):
        """Return the current alpha"""
        if self._wait > 0:
            return self._alpha_while_waiting
        return self._alpha

    def _update(self, t, alpha, step, final):
        raise NotImplementedError('I need an update rule')

    def update(self):
        if self._wait > 0:      # we are still waiting
            self._wait -= 1     # decrease the waiting time and keep waiting
            return self._alpha_while_waiting  # and keep waiting
        # We are done waiting, now we start counting
        self._counter += 1
        if self._counter < self._nb_updates:  # not enough updates
            return self._alpha
        else:  # enough updates, we are ready to reset the counter to zero
            self._counter = 0
        # and apply a step
        self._t += 1
        self._alpha = self._update(self._t, self._alpha, self._step, self._final)
        return self._alpha


class AnnealingSchedule(Schedule):
    """
    This class implements helper code for an annealing schedule.
    """

    def __init__(self, initial=1., final=1., step=0.,
                 wait=0, nb_updates=1, wait_value=None,
                 step_fn=lambda alpha, step, final: min(alpha + step, final),
                 name='alpha'):
        """

        :param initial:
        :param final:
        :param step:
        :param wait: how many updates should we wait before starting the schedule (the first step occurs after wait + nb_updates)
        :param nb_updates: number of updates between steps
        :param wait_value: a value (other than initial) to use while waiting
        :param step_fn: control a step in the schedule
            - the default step is additive and capped by `final`
            - one can design multiplicative steps
            - once can even make it a decreasing schedule
        :param name:
        """
        super(AnnealingSchedule, self).__init__(
            initial=initial,
            final=final,
            step=step,
            wait=wait,
            nb_updates=nb_updates,
            wait_value=wait_value,
            name=name
        )
        self._step_fn = step_fn

    def _update(self, t, alpha, step, final):
        return self._step_fn(alpha, step, final)


class _AnnealingSchedule(Schedule):
    """
    This class implements helper code for an annealing schedule.
    """

    def __init__(self, initial=1., final=1., step=0.,
                 wait=0, nb_updates=1, wait_value=None,
                 step_fn=lambda alpha, step, final, t: min(alpha + step, final),
                 name='alpha'):
        """

        :param initial:
        :param final:
        :param step:
        :param wait: how many updates should we wait before starting the schedule (the first step occurs after wait + nb_updates)
        :param nb_updates: number of updates between steps
        :param wait_value: a value (other than initial) to use while waiting
        :param step_fn: control a step in the schedule
            - the default step is additive and capped by `final`
            - one can design multiplicative steps
            - once can even make it a decreasing schedule
        :param name:
        """
        self._initial = initial
        self._final = final
        self._step = step
        self._wait = wait
        self._nb_updates = nb_updates
        self._alpha = initial
        self._alpha_while_waiting = wait_value if wait_value is not None else initial
        self._step_fn = step_fn
        self._counter = 0
        self._t = 0
        self.name = name

    def alpha(self):
        """Return the current alpha"""
        if self._wait > 0:
            return self._alpha_while_waiting
        return self._alpha

    def update(self):
        """
        Update schedule or waiting time.

        :param eoe: End-Of-Epoch flag
        :return: current alpha
        """
        if self._wait > 0:      # we are still waiting
            self._wait -= 1     # decrease the waiting time and keep waiting
            return self._alpha_while_waiting  # and keep waiting
        # We are done waiting, now we start counting
        self._counter += 1
        if self._counter < self._nb_updates:  # not enough updates
            return self._alpha
        else:  # enough updates, we are ready to reset the counter to zero 
            self._counter = 0
        # and apply a step
        self._t += 1
        self._alpha = self._step_fn(self._alpha, self._step, self._final, self._t)
        return self._alpha


class SigmoidSchedule(Schedule):
    """
    This class implements helper code for an annealing schedule.
    """

    def __init__(self, target,
                 growth_rate=1. / 100,
                 nu=1.,
                 nb_updates=1,
                 round=0.99,
                 lowerbound=0.0001,
                 name='alpha'):
        """

        :param target: the sigmoid will hit 1. roughly at this point
        :param growth_rate:
        :param nu: bigger numbers will make it less steep
        :param nb_updates: frequency of updates
        :param round: round to 1 whenever we reach this value
        :param name:
        """
        super(SigmoidSchedule, self).__init__(
            initial=1.,
            final=1.,
            step=0.,
            wait=0,
            nb_updates=nb_updates,
            wait_value=None,
            name=name
        )
        self._target = target
        self._growth_rate = growth_rate
        self._nu = nu
        self._round = round
        self._lowerbound = lowerbound

    def _update(self, t, alpha, step, final):
        alpha = generalised_logistic(
            t,
            lower_asymp=0.,  # we anneal from 0
            upper_asymp=1.,  # we anneal to 1
            growth_rate=self._growth_rate,
            starting_time=self._target,
            nu=self._nu,
            q=1.,
            c=1.,
        )
        if self._round is not None and alpha >= self._round:
            return 1.
        if alpha < self._lowerbound:
            return self._lowerbound
        return alpha
