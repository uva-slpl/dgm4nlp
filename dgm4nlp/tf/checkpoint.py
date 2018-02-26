import logging
from collections import namedtuple
import tensorflow as tf


class ModelTracker:
    """
    Use this helper to save the best model according to different criteria.
    """

    def __init__(self, session, model, output_dir):
        self._data = dict()
        self.session = session
        self.model = model
        self.output_dir = output_dir

    def _save(self, data, step, name, value):
        data.value = value
        data.step = step

        data.path = self.model.save(
            self.session,
            path="%s/model.best.%s.ckpt" % (self.output_dir, name)
        )

    def register(self, step, name, value, asc=True):
        """

        :param step: in which step the current parameters were obtained
        :param name: name of criterion (e.g. 'val_loss')
        :param value: value of criterion
        :param asc: whether this is a gain (True) or loss (False)
        :return: saved or not
        """
        data = self._data.get(name, None)
        if data is None:
            data = namedtuple('MetricData', ['value', 'step', 'path'])
            logging.info('Saving model: criterion=%s value=%f', name, value)
            self._save(data, step, name, value)
            self._data[name] = data
            return True
        else:
            if (asc and value >= data.value) or (not asc and value <= data.value):
                if step > data.step:
                    logging.info(
                        'Replacing model: criterion=%s old-value=%f new-value=%f old-step=%d new-step=%d',
                        name, data.value, value, data.step, step)
                    self._save(data, step, name, value)
                    return True
        return False

    def __iter__(self):
        return iter(self._data.items())