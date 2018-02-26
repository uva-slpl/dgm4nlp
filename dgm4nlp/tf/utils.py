"""
:Authors: - Wilker Aziz
"""
import tensorflow as tf
import re
#from dgm4nlp.tf._layers import dense


_STR_TO_FUNC = {
    'relu': tf.nn.relu,
    'tanh': tf.tanh,
    'softplus': tf.nn.softplus,
    'softmax': tf.nn.softmax,
    'sigmoid': tf.nn.sigmoid,
    'linear': tf.identity,
    'identity': tf.identity,
    None: tf.identity,
}


def get_nonlinearity(string):
    if string is None:
        return tf.identity
    if type(string) is str:
        try:
            return _STR_TO_FUNC[string]
        except KeyError:
            raise ValueError('Unknown function: %s' % string)
    elif type(tf.identity).__name__ == 'function':
        return string


def get_function_name(tf_function):
    if tf_function is None:
        return 'identity'
    if type(tf_function) is str:
        return tf_function
    return tf_function.__name__


def get_binary_operator(string):
    if string is None:
        return None
    string = string.lower()
    if string in ['none', '']:
        return None
    if string in ['add', 'sum']:
        return tf.add
    if string == ['multiply', 'times']:
        return tf.multiply
    raise ValueError('Unknown binary operator')


def add_uniform_noise(tensor, mask=None, minvalue=0., maxvalue=1.):
    """
    If you plot a histogram (for example with tensorboard) based on a tensor where some steps should have been
    masked out, you wil noticed a skew towards zero (in case your tensor evaluates to 0. at masked positions).
    This is a rather annoying behaviour! If your tensor values are bounded (such as for a cpd), you can circumvent
    that by adding uniform noise to your tensor before making the histogram.
     For unbounded tensors you can use Gaussian noise around the mean.

    :param tensor:
    :param mask: if provided it should have the same shape as the tensor
    :param minvalue:
    :param maxvalue:
    :return:
    """

    condition = tf.greater(tensor, minvalue) if mask is None else tf.cast(mask, tf.bool)
    return tf.where(
        condition=condition,
        x=tensor,
        y=tf.random_uniform(tf.shape(tensor), minvalue, maxvalue)
    )


def make_2d_mask(mask1, mask2):
    """

    :param mask1: [B, T1]
    :param mask2: [B, T2]
    :return: [B, T1, T2]
    """
    return tf.logical_and(
        tf.expand_dims(tf.cast(mask1, tf.bool), 2),  # [B, N, 1]
        tf.expand_dims(tf.cast(mask2, tf.bool), 1)
    )


def expand_into_2d_mask(repetitions, mask, dtype=tf.bool):
    """

    :param repetitions: N
    :param mask: [B, M]
    :param dtype: output type
    :return: [B, N, M]
    """
    valid = tf.cast(mask, dtype=dtype)
    # [B, N, M]
    valid = tf.tile(tf.expand_dims(valid, 1), [1, repetitions, 1])
    return valid


def mask2d(tensor, mask1, mask2, blank=0.):
    """

    :param tensor: [B, T1, T2]
    :param mask1: [B, T1]
    :param mask2: [B, T2]
    :param blank: blank
    :return: [B, T1, T2]
    """
    mask = tf.logical_and(
        tf.expand_dims(tf.cast(mask1, tf.bool), 2),  # [B, N, 1]
        tf.expand_dims(tf.cast(mask2, tf.bool), 1)
    )
    return tf.where(
        condition=mask,
        x=tensor,
        y=tf.fill(tf.shape(tensor), blank)
    )


def flag_argmax(tensor, dtype=tf.bool):
    return tf.cast(tf.equal(tensor, tf.reduce_max(tensor, axis=-1, keep_dims=True)), dtype)


def make_image(tensor):
    return tf.expand_dims(tensor, -1) * 255


def masked_softmax(logits, mask, paranoid=False):
    logits = tf.where(
        condition=tf.cast(mask, tf.bool),
        x=logits,
        y=tf.fill(tf.shape(logits), float('-inf'))
    )
    cpd = tf.nn.softmax(logits)
    if paranoid:
        cpd = tf.where(
            condition=tf.cast(mask, tf.bool),
            x=cpd,
            y=tf.random_uniform(tf.shape(cpd), 0., 1.)
        )
    return cpd


def dense(inputs, units, dropout=None, is_training=None, name=None, tag='ff_dropout', **kwargs):
    if dropout:  # with dropout configure
        if is_training is None:  # we require a training flag
            raise ValueError('With dropout I require a training flag (rank-0 boolean tensor)')
        if dropout > 0.:  # and only do anything if the drop rate is bigger than 0.
            with tf.variable_scope(tag if name is None else '{}/{}'.format(name, tag)):
                # variational dropout: we drop rows of the kernel
                #  which is equivalent to dropping units of the input
                inputs = tf.layers.dropout(
                    inputs=inputs,
                    rate=dropout,
                    training=is_training
                )
                return tf.layers.dense(inputs, units, **kwargs)
    return tf.layers.dense(inputs, units, **kwargs)


def _get_embedding_matrix(vocab_size, units, initializer_option, name):
    """

    :param initializer_option:
        - he/relu: He et al, good before tanh layers, based on normal distribution and fan_in
        - glorot/xavier/tanh: Glorot and Bengio, good before tanh layers, based on uniform dist and fan_avg
        - uniform/botev/css: uniform initializer, good for CSS layers
        - gaussian/normal: normal initializer
    :return:
    """
    if initializer_option in ['he', 'relu']:  # good for relu
        initializer = tf.variance_scaling_initializer(scale=2., mode='fan_in', distribution='normal')
    elif initializer_option in ['glorot', 'xavier', 'tanh']:  # good for tanh
        initializer = tf.variance_scaling_initializer(scale=1., mode='fan_avg', distribution='uniform')
    elif initializer_option in ['uniform', 'botev', 'css']:
        initializer = tf.random_uniform_initializer()
    elif initializer_option in ['bastings']:
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
    elif initializer_option in ['gaussian', 'normal']:
        initializer = tf.random_normal_initializer()
    else:
        raise ValueError('Unknown initializer')
    # [V, d]
    embedding_matrix = tf.get_variable(
        name=name,
        initializer=initializer,
        shape=[vocab_size, units]
    )
    return embedding_matrix


def get_embedding_matrix(vocab_size, units, initializer_option, dropout=None, is_training=None, dropout_tag='word_dropout', name='E', reuse=None):
    """
    Embedding matrix with optional variational word dropout.

    :param vocab_size:
    :param units:
    :param initializer_option:
    :param dropout:
    :param is_training:
    :param name:
    :param reuse:
    :return: emb_matrix, noisy_emb_matrix
    """
    if dropout:
        scope_name = dropout_tag
        if is_training is None:
            raise ValueError('With dropout I require a training flag (rank-0 boolean tensor)')
    else:
        scope_name = 'embeddings'

    with tf.variable_scope(scope_name, reuse=reuse):
        # [vocab_size, units]
        matrix = _get_embedding_matrix(vocab_size, units, initializer_option, name)
        noisy_matrix = matrix
        if dropout:  # apply word dropout if applicable
            noisy_matrix = tf.layers.dropout(
                inputs=matrix,
                rate=dropout,
                noise_shape=[vocab_size, 1],
                training=is_training,
            )
    return matrix, noisy_matrix


def get_timing_signal(length,
                      min_timescale=1,
                      max_timescale=1e4,
                      num_timescales=16):
    """Create Tensor of sinusoids of different frequencies.
    Source:
      https://github.com/tensorflow/tensor2tensor
    Args:
      length: Length of the Tensor to create, i.e. Number of steps.
      min_timescale: a float
      max_timescale: a float
      num_timescales: an int
    Returns:
      Tensor of shape (length, 2*num_timescales)
    """
    positions = tf.to_float(tf.range(length))
    log_timescale_increment = (tf.log(max_timescale / min_timescale) /
                               (num_timescales - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(positions, 1) * tf.expand_dims(inv_timescales, 0)
    return tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)


def add_timing_signal(x, min_timescale=1, max_timescale=1e4, num_timescales=16):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.
    This allows attention to learn to use absolute and relative positions.
    The timing signal should be added to some precursor of both the source
    and the target of the attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the depth dimension, padded with zeros to be the same depth as the input,
    and added into input.
    Source:
      https://github.com/tensorflow/tensor2tensor
    Args:
      x: a Tensor with shape [?, length, ?, depth]
      min_timescale: a float
      max_timescale: a float
      num_timescales: an int <= depth/2
    Returns:
      a Tensor the same shape as x.
    """
    length = tf.shape(x)[1]
    depth = tf.shape(x)[3]
    signal = get_timing_signal(length, min_timescale, max_timescale,
                               num_timescales)
    padded_signal = tf.pad(signal, [[0, 0], [0, depth - 2 * num_timescales]])
    return x + tf.reshape(padded_signal, [1, length, 1, depth])


def find_tagged_variables(tag='rnn_variational_dropout'):
    """
    Return kernels and biases trainable variables whose name match a tag.
    :param tag:
    :return: all vars with tag, kernels with tag, biases with tag
    """
    vars_with_dropout = [v for v in tf.trainable_variables() if tag in v.name]
    kernel_pattern = re.compile(r'kernel:[0-9]+$')
    kernels_with_dropout = [v for v in vars_with_dropout if kernel_pattern.search(v.name) is not None]
    bias_pattern = re.compile(r'bias:[0-9]+$')
    biases_with_dropout = [v for v in vars_with_dropout if bias_pattern.search(v.name) is not None]
    return vars_with_dropout, kernels_with_dropout, biases_with_dropout


def maximum(x, y, alpha=None, eps=1e-9):
    """

    :param x:
    :param y:
    :param alpha: 0 for average, large positive for maximum, large negative for minimum
        defaults to None which implies  tf.maximum
    :param eps:
    :return: tf.maximum if alpha is None else a smooth approximation
    """
    if alpha is None:
        return tf.maximum(x, y)
    # [..., 1]
    x = tf.expand_dims(x, -1)
    # [..., 1]
    y = tf.expand_dims(y, -1)
    # [..., 2]
    z = tf.concat([x, y], -1)
    weights = tf.nn.softmax(z * alpha + eps)
    # [...]
    return tf.reduce_sum(z * weights, -1)


def hinge(x, margin=None, alpha=None):
    """
    Hinge function

    :param x:
    :param margin: defaults to 1.
    :param alpha: defaults to None which implies tf.maximum for hinge
        if alpha is a positive scalar, we replace tf.maximum by a smooth approximation
    :return:
    """
    if margin is None:
        margin = tf.ones_like(x)
    return maximum(tf.zeros_like(x), margin - x, alpha=alpha)


class Bound:

    def __init__(self, bound=None, alpha=2., lowerbound=True):
        self._bound = bound
        self._alpha = alpha if lowerbound else -alpha
        if alpha < 0.:
            raise ValueError('Alpha must be non-negative. Consider switching the lowerbound flag.')

    def bound(self, x):
        if self._bound is None:
            return x
        else:
            return maximum(x, tf.ones_like(x) * self._bound, alpha=self._alpha)

    def __call__(self, x):
        return self.bound(x)
