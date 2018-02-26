import logging
from collections import namedtuple
from collections import defaultdict
import tensorflow as tf
from tabulate import tabulate


class ModelTracker:
    """
    Use this helper to save the best model according to different criteria.
    """

    def __init__(self, session, model, output_dir):
        self._data = dict()
        self.session = session
        self.model = model
        self.output_dir = output_dir
        self.records = defaultdict(list)
        self.names = set()
        self.stepbystep = defaultdict(dict)

    def _save(self, data, step, name, value, saving=True):
        data.value = value
        data.step = step

        data.path = self.model.save(
            self.session,
            path="%s/model.best.%s.ckpt" % (self.output_dir, name)
        )

    def get_value(self, monitor):
        record = self._data.get(monitor, None)
        if record is None:
            return None
        return record.value

    def register(self, step, name, value, asc=True, saving=True):
        """

        :param step: in which step the current parameters were obtained
        :param name: name of criterion (e.g. 'val_loss')
        :param value: value of criterion
        :param asc: whether this is a gain (True) or loss (False)
        :return: saved or not
        """
        # general logging
        self.stepbystep[step][name] = value
        self.names.add(name)
        # tracking
        data = self._data.get(name, None)
        if data is None:
            data = namedtuple('MetricData', ['value', 'step', 'path'])
            logging.info('Saving model: criterion=%s value=%f', name, value)
            self._save(data, step, name, value, saving=saving)
            self._data[name] = data
            return True
        else:
            if (asc and value >= data.value) or (not asc and value <= data.value):
                if step > data.step:
                    logging.info(
                        'Replacing model: criterion=%s old-value=%f new-value=%f old-step=%d new-step=%d',
                        name, data.value, value, data.step, step)
                    self._save(data, step, name, value, saving=saving)
                    return True
        return False

    def __iter__(self):
        return iter(self._data.items())

    def log_steps(self, ostream):
        names = list(sorted(self.names))
        rows = []
        for step, records in sorted(self.stepbystep.items(), key=lambda pair: pair[0]):
            rows.append([step] + [records.get(name, '') for name in names])
        print(tabulate(rows, headers=['step'] + names), file=ostream)


class ConvergenceMonitor:

    @staticmethod
    def preprocess_number(n):
        if n is None:
            return None
        if n < 0:
            return None
        return n

    def __init__(self,
                 objective=None,
                 sensitivity=0.1,
                 relative=False,
                 patience=10,
                 ckpt_interval=1000,
                 asc=False,
                 min_steps=None,
                 max_steps=None,
                 min_epochs=1,
                 max_epochs=100):
        """

        :param objective: use None for convergence on number of epochs, otherwise provide the name of an objective
            e.g. validation.objective
        :param sensitivity: minimum improvement
        :param patience: maximum number of updates without a minimum improvement
        :param asc: minimisation (False) or maximisation (True)
        :param min_steps: we start reducing patience when both min_steps and min_epochs have been met
        :param max_steps: we stop when either max_steps or max_epochs have been met
        :param min_epochs: number of epochs before tracking the objective
        :param max_epochs: maximum number of epochs (regardless of objective)
        """
        self.min_epochs = ConvergenceMonitor.preprocess_number(min_epochs)
        self.max_epochs = ConvergenceMonitor.preprocess_number(max_epochs)
        self.min_steps = ConvergenceMonitor.preprocess_number(min_steps)
        self.max_steps = ConvergenceMonitor.preprocess_number(max_steps)
        self.objective = objective
        self.sensitivity = sensitivity
        self.relative = relative
        self.patience = patience
        self.ckpt_interval = ckpt_interval
        self.asc = asc
        self._record = None
        self._epochs = 0
        self._steps = 0

    @property
    def epochs(self):
        return self._epochs

    def reached_minimum(self):
        if self.min_steps is None and self.min_epochs is None:  # we do not have a minimum
            return True
        elif self.min_steps is None:  # we are counting epochs
            return self._epochs > self.min_epochs
        elif self.min_epochs is None:  # we are counting steps
            return self._steps > self.min_steps
        else:  # we need to satisfy both criteria
            return self._epochs > self.min_epochs and self._steps > self.min_steps

    def reached_maximum(self):
        if self.max_steps is None and self.max_epochs is None:  # we do not have a maximum
            return False
        elif self.max_steps is None:  # we are counting epochs
            return self._epochs > self.max_epochs
        elif self.max_epochs is None:  # we are counting steps
            return self._steps > self.max_steps
        else:  # we need to satisfy either criteria
            return self._epochs > self.max_epochs or self._steps > self.max_steps

    def target_value(self, best_value):
        if self.relative:  # relative/multiplicative improvements
            if self.asc:
                return best_value * (1. + self.sensitivity)
            else:
                return best_value * (1. - self.sensitivity)
        else:  # absolute/additive improvements
            if self.asc:
                return best_value + self.sensitivity
            else:
                return best_value - self.sensitivity

    def update(self, objectives: dict):

        if not self.objective or self.objective.lower() == 'none':  # no early stopping configured
            return

        if not self.reached_minimum():  # no updates until we reach the minimum number of epochs
            return

        if self.converged():  # no updates if we have converged
            return

        try:
            value = objectives[self.objective]
        except ValueError:
            raise ValueError('You must provide %s' % self.objective)

        if self._record is None:
            self._record = [value, self.patience]
            target_value = self.target_value(value)
            logging.info('Convergence: waiting for %s to improve from %.4f to %.4f with patience %d', self.objective,
                         self._record[0], target_value, self._record[1])
        else:
            best_value = self._record[0]
            target_value = self.target_value(best_value)

            if (self.asc and value >= target_value) or (not self.asc and value <= target_value):
                # update the record upon improvement
                self._record = [value, self.patience]  # reset patience
                logging.info('Convergence: %s with patience %d improved from %.4f to %.4f.', self.objective, self._record[1], best_value, value)
            else:
                self._record[1] -= 1  # decrease patience
                logging.info('Convergence: %s with (reduced) patience %d did not reach the target %.4f', self.objective, self._record[1], target_value)

    def take_another_step(self):
        self._steps += 1
        return not self.converged()

    def start_another_epoch(self):
        """Starts a new epoch and returns the result of the convergence test"""
        self._epochs += 1
        return not self.converged()

    def converged(self):
        if self.reached_maximum():
            logging.info('Convergence: reached maximum number of steps %r/%r or epochs %r/%r',
                         self._steps, self.max_steps,
                         self._epochs, self.max_epochs)
            return True

        if not self.reached_minimum():  # Here we know we haven't surpassed the minimum
            return False

        if not self.objective or self.objective.lower() == 'none':  # not tracking an objective
            return False

        elif self._record is not None:  # waiting for an objective to converge and have information about it
            if self._record[1] <= 0:  # run out of patience
                logging.info('Convergence: ran out of patience for %s', self.objective)
                return True

        return False
