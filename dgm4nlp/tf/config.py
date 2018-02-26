"""
Configuration objects such as

    - CSS
    - Dropout


:Authors: - Wilker Aziz
"""
from dgm4nlp.tf.utils import get_nonlinearity, get_function_name


class CSSConfig:

    def __init__(self,
                 nb_softmax_samples=1000,
                 use_pre_css_layer=False,
                 pre_css_activation=None,
                 share_class_embeddings=True,
                 allow_dropout=True,
                 softmax_q_inv_temperature=None
                 ):
        self.nb_softmax_samples = nb_softmax_samples
        self.use_pre_css_layer = use_pre_css_layer
        self.pre_css_activation = get_nonlinearity(pre_css_activation)
        self.share_class_embeddings = share_class_embeddings
        self.allow_dropout = allow_dropout
        self.softmax_q_inv_temperature = softmax_q_inv_temperature

    def __repr__(self):
        return '%s(nb_softmax_samples=%r, use_pre_css_layer=%r, pre_css_activation=%r, share_class_embeddings=%r, allow_dropout=%r, softmax_q_inv_temperature=%r)' % (
            self.__class__.__name__,
            self.nb_softmax_samples,
            self.use_pre_css_layer,
            get_function_name(self.pre_css_activation),
            self.share_class_embeddings,
            self.allow_dropout,
            self.softmax_q_inv_temperature
        )


class DropoutConfig:

    def __init__(self,
                 rnn_dropout=0.,
                 ff_dropout=0.,
                 word_dropout=0.,
                 rnn_l2='auto',
                 ff_l2='auto',
                 word_l2='auto',
                 variational_recurrent=True):
        self.rnn_drop_rate = rnn_dropout
        self.ff_drop_rate = ff_dropout
        self.word_drop_rate = word_dropout
        self.variational_recurrent = variational_recurrent
        # base option
        self._base_rnn_l2 = rnn_l2
        self._base_ff_l2 = ff_l2
        self._base_word_l2 = word_l2
        # to be set
        self._ff_l2 = None
        self._word_l2 = None
        self._rnn_l2 = None

    def __repr__(self):
        return '%s(rnn_dropout=%r, ff_dropout=%r, word_dropout=%r, rnn_l2=%r, ff_l2=%r, word_l2=%r, variational_recurrent=%r)' % (
            self.__class__.__name__,
            self.rnn_drop_rate,
            self.ff_drop_rate,
            self.word_drop_rate,
            self._base_rnn_l2,
            self._base_ff_l2,
            self._base_word_l2,
            self.variational_recurrent)

    @property
    def rnn_dropout_l2(self):
        if self._rnn_l2 is None:
            raise ValueError('Perhaps you forgot to config rnn-dropout L2')
        return self._rnn_l2

    @property
    def ff_dropout_l2(self):
        if self._ff_l2 is None:
            raise ValueError('Perhaps you forgot to config ff-dropout L2')
        return self._ff_l2

    @property
    def word_dropout_l2(self):
        if self._word_l2 is None:
            raise ValueError('Perhaps you forgot to config word-dropout L2')
        return self._word_l2

    def config(self, training_size):
        if self._base_ff_l2 is 'auto':
            self._ff_l2 = [(1. - self.ff_drop_rate) / training_size, 1. / training_size]
        elif type(self._base_ff_l2) in [float, int]:  # here we use the given strength as a base and scale by keep rate
            self._ff_l2 = [(1. - self.ff_drop_rate) * self._base_ff_l2, 1. * self._base_ff_l2]
        else:
            raise ValueError('I cannot deal with word-dropout l2 option')

        if self._base_word_l2 is 'auto':
            self._word_l2 = (1. - self.word_drop_rate) / training_size
        elif type(self._base_word_l2) in [float, int]:  # the embedding matrix is a kernel
            self._word_l2 *= (1. - self.word_drop_rate)  # so we scale by keep rate
        else:
            raise ValueError('I cannot deal with word-dropout l2 option')

        if self._base_rnn_l2 is 'auto':
            self._rnn_l2 = [(1. - self.rnn_drop_rate) / training_size, 1. / training_size]
        elif type(self._base_rnn_l2) in [float, int]:
            self._rnn_l2 = [(1. - self.rnn_drop_rate) * self._base_rnn_l2, 1. * self._base_rnn_l2]
        else:
            raise ValueError('I cannot deal with rnn-dropout l2 option')

