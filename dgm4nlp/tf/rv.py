"""
:Authors: - Wilker Aziz
"""
import tensorflow as tf
import dgm4nlp.tf.dist as dist
import logging
import numpy as np
from dgm4nlp.tf.utils import dense, get_function_name, make_image


def pprint_activation(tf_function, arg_string):
    """Pretty print the activation as a function of something"""
    if tf_function in [None, tf.identity]:
        return arg_string
    else:
        return '{}({})'.format(get_function_name(tf_function), arg_string)


def delta_scalar(var_name, initializer, activation=tf.identity, summary=False):
    """Returns a scalar trainable parameter"""
    logging.info('  %s = scalar(%s) with s a parameter', var_name, pprint_activation(activation, 's'))
    with tf.variable_scope(var_name):
        outputs = activation(tf.get_variable(
            name='pre_s',
            shape=[] if initializer is None else None,
            dtype=tf.float32,
            initializer=initializer)
        )
        if summary:
            tf.summary.scalar('parameter', outputs)
    return outputs


def delta_tensor(shape, inputs, var_name, activation=tf.identity, initializer=None, summary=False, dropout=0., is_training=None):
    """
    Returns a tensor filled with `dim` parameters or predictions.
    :param shape: [batch_size, max_time, dim]
    :param inputs: if not None, then we predict a tensor [batch_size, max_time, dim]
        otherwise we tile `dim` parameters to make a [batch_size, max_time, dim] tensor
    :param var_name:
    :param activation:
    :param summary:
    :param dropout:
    :param is_training:
    :return:
    """
    dim = shape[-1]
    if inputs is None:
        logging.info('  %s = tensor(%s) with t a parameter', var_name, pprint_activation(activation, 't'))
        # here we have `dim` parameters
        with tf.variable_scope(var_name):
            outputs = activation(tf.get_variable(
                name='pre_t',
                shape=dim if initializer is None else None,
                dtype=tf.float32,
                initializer=initializer
            ))
            expanded_shape = [1] * len(shape)
            expanded_shape[-1] = dim  # e.g. [1, 1, dim]
            copies = list(shape)
            copies[-1] = 1  # e.g. [B, T, 1]
            outputs = tf.tile(tf.reshape(outputs, expanded_shape), copies)
            if summary:
                tf.summary.histogram('parameter', outputs)
    else:
        logging.info('  %s = tensor(%s) with t a prediction', var_name, pprint_activation(activation, 't'))
        # here we predict a tensor with shape `shape`
        with tf.variable_scope(var_name):
            outputs = dense(
                inputs=inputs,
                units=dim,
                activation=activation,
                dropout=dropout,
                is_training=is_training
            )
            if summary:
                tf.summary.histogram('prediction', outputs)
    return outputs


def delta_cholesky(shape, inputs, var_name, summary=False, dropout=0., is_training=None):
    """
    A Cholesky factor (parameter or prediction).
    :param shape: [B, T, d]
    :param inputs: if None, then we return a Cholesky trainable parameter for a d-by-d covariance and tile it
        to the given shape [B, T, d].
        Otherwise we predict a Cholesky factor for a d-by-d covariance.
    :param var_name:
    :param summary:
    :param dropout:
    :param is_training:
    :return:
    """
    logging.info('  %s = cholesky(diag, off-diag)', var_name)
    with tf.variable_scope(var_name):
        dim = shape[-1]
        # [B, N, dim]
        diag = delta_tensor(
            shape=shape,
            inputs=inputs,  # we delegate the decision of parameterising or predicting
            var_name='diag',
            activation=tf.nn.softplus,
            summary=False,
            dropout=dropout,
            is_training=is_training
        )
        # [B, N, dim * dim]
        off_diag = delta_tensor(
            shape=shape[:-1] + [dim * dim],
            inputs=inputs,  # we delegate the decision of parameterising or predicting
            var_name='off-diag',
            activation=tf.identity,
            summary=False,
            dropout=dropout,
            is_training=is_training
        )
        # [B, N, dim, dim]
        off_diag = tf.reshape(off_diag, shape[:-1] + [dim, dim])
        # here we replace the diagonal with non-negative reals
        _chol = tf.matrix_set_diag(off_diag, diag)
        # here we zero out elements above the diagonal
        chol = tf.matrix_band_part(_chol, -1, 0)
        if summary:
            sum_name = 'parameter' if inputs is None else 'prediction'
            tf.summary.histogram(sum_name, _chol)  # we use _chol for the histrogram to avoid skewing towards 0.
            tf.summary.image('mean-{}'.format(sum_name), make_image(tf.reduce_mean(chol, 1, keep_dims=True)))
        return chol


class RV:

    def kl(self, other: 'RV'):
        pass

    def mean(self):
        pass


class CustomRV:

    def __init__(self, value, parameters: list, kl_func, name='Custom'):
        self.value = value
        self._parameters = parameters
        self._kl_func = kl_func
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.value

    def kl(self, other: 'CustomRV'):
        if not isinstance(other, CustomRV):
            raise ValueError('I expected another CustomRV: KL[self || other]')
        return self._kl_func(other._parameters)


class Delta(RV):
    """
    A distribution that is degenerate at a certain value.
    This value may be a parameter or a prediction.

    """
    def __init__(self, shape, parameter, summary=False,  name='Delta'):
        self.shape = shape
        self.name = name
        self.summary = summary
        self.value = parameter
        if summary:
            tf.summary.histogram('sample', self.value)

    def mean(self):
        return self.value

    @staticmethod
    def construct_tensor(
            shape, inputs=None, activation=tf.identity,
            summary=False, dropout=0., is_training=None, name='Delta') -> 'Delta':
        logging.info(' %s ~ Delta(d)', name)
        with tf.variable_scope(name):
            parameter = delta_tensor(
                shape=shape,
                inputs=inputs,
                activation=activation,
                var_name='d',
                summary=summary,
                dropout=dropout,
                is_training=is_training
            )
            return Delta(
                shape=shape,
                parameter=parameter,
                summary=summary,
                name=name
            )

    @staticmethod
    def construct_constant(shape, constant: float, summary=False, name='Delta') -> 'Delta':
        logging.info(' %s ~ Delta(%s)', name, constant)
        # here we have `dim` parameters
        with tf.variable_scope(name):
            expanded_shape = [1] * len(shape)
            expanded_shape[-1] = shape[-1]  # e.g. [1, 1, dim]
            copies = list(shape)
            copies[-1] = 1  # e.g. [B, T, 1]
            parameter = tf.tile(tf.reshape(constant, expanded_shape), copies)
            if summary:
                tf.summary.histogram('constant', parameter)
            return Delta(
                shape=shape,
                parameter=parameter,
                summary=summary,
                name=name
            )

    def __call__(self, *args, **kwargs):
        return self.value

    def kl(self, other: 'Delta'):
        if not isinstance(other, Delta):
            raise ValueError('%s expected another delta distribution KL[self || other] but got %s' % (self.name, type(other)))
        return None


class Normal(RV):

    @classmethod
    def construct(cls, shape, inputs, mc=None, summary=False, dropout=0., is_training=None, name='Normal') -> 'Normal':
        pass


class NormalCholesky(Normal):

    def __init__(self, shape, loc, scale_tril, mc=None, summary=False, name='Normal'):
        self.shape = shape
        self.name = name
        self._loc = loc
        self._scale_tril = scale_tril
        self._tf_normal = tf.contrib.distributions.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril)

        if mc is None:
            self.value = self._tf_normal.sample()
        else:
            self.value = tf.cond(
                mc,
                true_fn=lambda: self._tf_normal.sample(),
                false_fn=lambda: self._loc
            )

        if summary:
            tf.summary.histogram('sample', self.value)

    def mean(self):
        return self._loc

    def __call__(self, *args, **kwargs):
        return self.value

    def kl(self, other: 'NormalCholesky'):
        if not isinstance(other, NormalCholesky):
            raise ValueError('I expected another normal distribution but got %s' % type(other))
        return tf.contrib.distributions.kl_divergence(self._tf_normal, other._tf_normal)

    @classmethod
    def construct(cls, shape, inputs, mc=None, summary=False, dropout=0., is_training=None, name='NormalCholesky') -> 'NormalCholesky':

        if shape[-1] == 1:
            raise ValueError('Use Normal for univariates')

        logging.info(' %s ~ NormalCholesky(loc, scale)', name)

        with tf.variable_scope(name):
            loc = delta_tensor(
                shape=shape,
                inputs=inputs,
                var_name='loc',
                activation=tf.identity,
                summary=summary,
                dropout=dropout,
                is_training=is_training
            )
            scale = delta_cholesky(
                shape=shape,
                inputs=inputs,
                var_name='scale',
                summary=summary,
                dropout=dropout,
                is_training=is_training
            )

            return NormalCholesky(
                shape=shape,
                loc=loc,
                scale_tril=scale,
                mc=mc,
                summary=summary,
                name=name
            )

    @classmethod
    def standard_normal(cls, shape, dtype=tf.float32, mc=None, summary=False, name='StdNormal') -> 'NormalCholesky':
        logging.info(' %s ~ NormalCholesky(0., 1.)', name)

        return NormalCholesky(
            shape=shape,
            loc=tf.zeros(shape, dtype=dtype),
            scale_tril=None,
            mc=mc,
            summary=summary,
            name=name
        )


class NormalDiag(Normal):

    def __init__(self, shape, loc, scale, mc=None, summary=False, name='Normal'):
        self.shape = shape
        self.name = name
        self._loc = loc
        self._scale = scale

        D = dist.Normal()

        if mc is None:
            self.value = D.random(shape, [loc, scale])
        else:
            self.value = tf.cond(
                mc,
                true_fn=lambda: D.random(self.shape, [self._loc, self._scale]),
                false_fn=lambda: D.mean([self._loc, self._scale])
            )
        if summary:
            tf.summary.histogram('sample', self.value)

    def params(self):
        return [self._loc, self._scale]

    def mean(self):
        return self._loc

    def __call__(self, *args, **kwargs):
        return self.value

    def kl(self, other: 'NormalDiag'):
        if not isinstance(other, NormalDiag):
            raise ValueError('I expected another normal distribution: KL[self || other]')
        location_i, scale_i = self._loc, self._scale  # [mean, std]
        location_j, scale_j = other._loc, other._scale  # [mean, std]
        var_i = scale_i ** 2
        var_j = scale_j ** 2
        term1 = 1 / (2 * var_j) * ((location_i - location_j) ** 2 + var_i - var_j)
        term2 = tf.log(scale_j) - tf.log(scale_i)
        return term1 + term2

    @staticmethod
    def standard_normal(shape, mc=None, summary=False, name='StdNormal') -> 'NormalDiag':
        logging.info(' %s ~ NormalDiag(0., 1.)', name)
        with tf.variable_scope(name):
            return NormalDiag(
                shape=shape,
                loc=0.,
                scale=1.,
                mc=mc,
                summary=summary,
                name=name
            )

    @classmethod
    def construct(cls, shape, inputs, mc=None, summary=False, dropout=0., is_training=None,
                  name='NormalDiag') -> 'NormalDiag':
        logging.info(' %s ~ NormalDiag(loc, scale)', name)
        with tf.variable_scope(name):
            loc = delta_tensor(
                shape=shape,
                inputs=inputs,
                var_name='loc',
                activation=tf.identity,
                summary=summary,
                dropout=dropout,
                is_training=is_training
            )

            scale = delta_tensor(
                shape=shape,
                inputs=inputs,
                var_name='scale',
                activation=tf.nn.softplus,
                summary=summary,
                dropout=dropout,
                is_training=is_training
            )

            return NormalDiag(
                shape=shape,
                loc=loc,
                scale=scale,
                mc=mc,
                summary=summary,
                name=name
            )


class TransformedNormal(RV):
    def __init__(self, shape, normal_rv: Normal, activation,
                 summary=False, name='TransformedNormal'):
        self.shape = shape
        self.name = name
        self.summary = summary
        self._activation = activation
        self._normal_rv = normal_rv
        self.value = self._activation(normal_rv.value)
        if self.summary:
            tf.summary.histogram('sample', self.value)

    @staticmethod
    def construct(
            shape, inputs, activation,
            normal_cls: Normal=NormalDiag,
            mc=None, summary=False, dropout=0., is_training=None,
            normal_name='N', name='TransformedNormal') -> 'TransformedNormal':
        logging.info(' %s = %s where %s ~ %s(loc, scale)',
                     name, pprint_activation(activation, normal_name),
                     normal_name, normal_cls.__name__)
        with tf.variable_scope(name):
            normal_rv = normal_cls.construct(
                shape=shape,
                inputs=inputs,
                mc=mc,
                summary=summary,
                dropout=dropout,
                is_training=is_training,
                name=normal_name)
            return TransformedNormal(
                shape=shape,
                normal_rv=normal_rv,
                activation=activation,
                summary=summary,
                name=name
            )

    def mean(self):
        return self._activation(self._normal_rv.mean())

    def __call__(self, *args, **kwargs):
        return self.value

    def kl(self, other: 'TransformedNormal'):
        if not isinstance(other, TransformedNormal):
            raise ValueError('I expected another (transformed) normal distribution: KL[self || other]')
        if self._activation is not other._activation:
            raise ValueError('You are comparing different transformations of normal variables')
        return self._normal_rv.kl(other._normal_rv)


class Kuma(RV):
    """
    About Kumaraswamy: https://en.wikipedia.org/wiki/Kumaraswamy_distribution


    In our version it is: Kuma(a + t1(k1, k2), b + t2(k1, k2))
      and the default t1 returns k1, and the default t2 returns k2
    * kuma biases (a, b) can prevent bimodal solutions, just set them to [1., 1.]
    * transformations (t1, t2) can enforce heavier tails, you can for example, use kuma_t2=tf.add
      to make the second parameter always greater or equal to first shape parameter
    * a and b are never predicted, at most they are parameters, to get learnable parameters set them to [None, None]
      in case you set them to None, then you probably want to choose an initilizer, for example
      kuma_biases_initializer=[1., 3.]  (mind that these numbers will go through a softplus)
    * k1 and k2 are parameters if there is not enc_pc otherwise they will be predictions
       in case you use parameters, you can configure an initializer, e.g. kuma_ks_initializer=[-0.5, -0.5]
       mind that the intializer also goes through a softplus and therefore this example gives you
       a distribution very similar to Kuma(0.5, 0.5) which is U-shaped (bimodal)

    """

    def __init__(self, shape, alpha, beta, num_terms=10,
                 mc=None, summary=False, name='Kuma'):
        """
        X ~ Kuma(a + t1(k1, k2), b + t2(k1, k2))
            where k1 = f(inputs) and k2 = g(inputs) if inputs are given
             otherwise k1 and k2 are parameters
            a and b can be set to 0. for no effect,
            they can be specified as tensors [B, N, 1] or scalars
            or they can be made learned parameters if set to None

            finally t1 and t2 gives one the chance to combine k1 and k2
            by default t1 returns k1 and t2 returns k2

        :param shape:
        :param inputs:
        :param a:
        :param b:
        :param t1: binary operator
        :param t2: binary operator
        :param summary:
        :param name:
        """
        self.shape = shape
        self.name = name
        self.summary = summary
        self._num_terms = num_terms
        self._alpha = alpha
        self._beta = beta

        D = dist.Kuma([0.5, 0.5])  # the parameters are irrelevant (TODO: change API)
        if mc is None:
            self.value = D.random(self.shape, [self._alpha, self._beta])
        else:
            self.value = tf.cond(
                mc,
                true_fn=lambda: D.random(self.shape, [self._alpha, self._beta]),
                false_fn=lambda: D.mean([self._alpha, self._beta])
            )
        if self.summary:
            tf.summary.histogram('alpha', alpha)
            tf.summary.histogram('beta', beta)
            tf.summary.histogram('sample', self.value)

    @staticmethod
    def construct(
            shape, inputs, a=0., b=0., t1=None, t2=None,
            a_initializer=None, b_initializer=None,
            k1_initializer=None, k2_initializer=None,
            mc=None, summary=False, dropout=0., is_training=None, num_terms=10, name='Kuma') -> 'Kuma':
        """
        X ~ Kuma(a + t1(k1, k2), b + t2(k1, k2))
            where k1 = f(inputs) and k2 = g(inputs) if inputs are given
             otherwise k1 and k2 are parameters
            a and b can be set to 0. for no effect,
            they can be specified as tensors [B, N, 1] or scalars
            or they can be made learned parameters if set to None

            finally t1 and t2 gives one the chance to combine k1 and k2
            by default t1 returns k1 and t2 returns k2

        :param shape:
        :param inputs:
        :param a:
        :param b:
        :param t1: binary operator
        :param t2: binary operator
        :param summary:
        :param name:
        """
        logging.info(' %s ~ Kuma(alpha, beta) where `alpha = a + %s`, `beta = b + %s`, `a` is %s and `b` is %s',
                     name,
                     'k1' if t1 is None else pprint_activation(t1, 'k1, k2'),
                     'k2' if t2 is None else pprint_activation(t2, 'k1, k2'),
                     'a parameter' if a is None else a,
                     'a parameter' if b is None else b)
        with tf.variable_scope(name):
            if a is None:
                a = delta_scalar(
                    var_name='a',
                    initializer=a_initializer,
                    activation=tf.nn.softplus,
                    summary=summary
                )
            if b is None:
                b = delta_scalar(
                    var_name='b',
                    initializer=b_initializer,
                    activation=tf.nn.softplus,
                    summary=summary
                )
            k1 = delta_tensor(
                shape=shape,
                inputs=inputs,
                var_name='k1',
                activation=tf.nn.softplus,
                initializer=k1_initializer,
                summary=summary,
                dropout=dropout,
                is_training=is_training
            )
            k2 = delta_tensor(
                shape=shape,
                inputs=inputs,
                var_name='k2',
                activation=tf.nn.softplus,
                initializer=k2_initializer,
                summary=summary,
                dropout=dropout,
                is_training=is_training
            )
            if t1 is None:
                alpha = a + k1
            else:
                alpha = a + t1(k1, k2)
            if t2 is None:
                beta = b + k2
            else:
                beta = b + t2(k1, k2)

            return Kuma(
                shape=shape,
                alpha=alpha,
                beta=beta,
                num_terms=num_terms,
                mc=mc,
                summary=summary,
                name=name)

    def __call__(self, *args, **kwargs):
        return self.value

    def kl(self, other: 'Kuma'):
        # TODO: Kuma || Kuma rather than Kuma || Beta
        if not isinstance(other, Kuma):
            raise ValueError('I expected another kuma distribution: KL[self || other]')
        kuma_a, kuma_b = self._alpha, self._beta
        beta_a, beta_b = other._alpha, other._beta
        term1 = (kuma_a - beta_a) / kuma_a * (- np.euler_gamma - tf.digamma(kuma_b) - 1.0 / kuma_b)
        term1 += tf.log(kuma_a * kuma_b) + tf.lbeta(
            tf.concat([tf.expand_dims(beta_a, -1), tf.expand_dims(beta_b, -1)], -1))
        term1 += - (kuma_b - 1) / kuma_b
        # Truncated Taylor expansion around 1
        taylor = tf.zeros(tf.shape(kuma_a))
        for m in range(1, self._num_terms + 1):  # m should start from 1 (otherwise betafn will be inf)!
            taylor += dist.tf_beta_fn(m / kuma_a, kuma_b) / (m + kuma_a * kuma_b)
        term2 = (beta_b - 1) * kuma_b * taylor
        return term1 + term2  # tf.maximum(0., term1 + term2)



def kuma_sample(shape, inputs, var_name, mc_fwd,
                a=0., b=0., t1=None, t2=None,
                stochastic=True, summary=False):
    """
    X ~ Kuma(a + t1(k1, k2), b + t2(k1, k2))
        where k1 = f(inputs) and k2 = g(inputs) if inputs are given
         otherwise k1 and k2 are parameters
        a and b can be set to 0. for no effect,
        they can be specified as tensors [B, N, 1] or scalars
        or they can be made learned parameters if set to None

        finally t1 and t2 gives one the chance to combine k1 and k2
        by default t1 returns k1 and t2 returns k2

    :param shape:
    :param inputs:
    :param var_name:
    :param mc_fwd:
    :param a:
    :param b:
    :param t1:
    :param t2:
    :param stochastic:
    :param summary:
    :return:
    """

    if not stochastic:
        return delta_tensor(
            shape=shape,
            inputs=inputs,
            var_name=var_name,
            activation=tf.nn.sigmoid,
            summary=summary)

    logging.info('%s ~ Kuma(a + alpha, b + beta)', var_name)

    with tf.variable_scope(var_name):
        if a is None:
            a = delta_scalar(
                var_name='a',
                initializer=-1.,
                activation=tf.nn.softplus,
                summary=summary
            )
        if b is None:
            b = delta_scalar(
                var_name='b',
                initializer=-1.,
                activation=tf.nn.softplus,
                summary=summary
            )
        k1 = delta_tensor(
            shape=shape,
            inputs=inputs,
            var_name='k1',
            activation=tf.nn.softplus,
            summary=summary
        )
        k2 = delta_tensor(
            shape=shape,
            inputs=inputs,
            var_name='k2',
            activation=tf.nn.softplus,
            summary=summary
        )
        if t1 is None:
            alpha = a + k1
        else:
            alpha = a + t1(k1, k2)
        if t2 is None:
            beta = b + k2
        else:
            beta = b + t2(k1, k2)
        D = dist.Kuma([0.5, 0.5])  # the parameters are irrelevant (TODO: change API)
        outputs = tf.cond(
            mc_fwd,
            true_fn=lambda: D.random(shape, [alpha, beta]),
            false_fn=lambda: D.mean([alpha, beta])
        )
    if summary:
        tf.summary.histogram(var_name, outputs)
    return outputs


def exp_sample(shape, inputs, var_name, mc_fwd, stochastic=True, summary=False):
    """
    A Kuma-distributed variable.

    :param inputs: [B, N, 1] for predictions or None for parameters
    :param shape: tuple([B, N, 1])
    :param var_name: name of the random variable
    :return: random draw (training) or mean (prediction)
    """
    if not stochastic:
        return delta_tensor(
            shape=shape,
            inputs=inputs,
            var_name=var_name,
            activation=tf.nn.softplus,
            summary=summary)

    logging.info('%s ~ Exp(r)', var_name)

    with tf.variable_scope(var_name):
        r = delta_tensor(
            shape=shape,
            inputs=inputs,
            var_name='r',
            activation=tf.nn.softplus,
            summary=summary
        )
        D = dist.Exponential()
        outputs = tf.cond(
            mc_fwd,
            true_fn=lambda: D.random(shape, [r]),
            false_fn=lambda: D.mean([r])
        )
    if summary:
        tf.summary.histogram(var_name, outputs)
    return outputs


def normal_sample(shape, inputs, var_name, mc_fwd, stochastic=True, summary=False):
    if not stochastic:
        return delta_tensor(
            shape=shape,
            inputs=inputs,
            var_name=var_name,
            activation=tf.identity,
            summary=summary
        )

    logging.info('%s ~ Normal(loc, scale)', var_name)

    with tf.variable_scope(var_name):
        loc = delta_tensor(
            shape=shape,
            inputs=inputs,
            var_name='loc',
            activation=tf.identity,
            summary=summary
        )
        scale = delta_tensor(
            shape=shape,
            inputs=inputs,
            var_name='scale',
            activation=tf.nn.softplus,
            summary=summary
        )
        D = dist.Normal()
        outputs = tf.cond(
            mc_fwd,
            true_fn=lambda: D.random(shape, [loc, scale]),
            false_fn=lambda: D.mean([loc, scale])
        )
    if summary:
        tf.summary.histogram(var_name, outputs)
    return outputs


def logit_normal_sample(shape, inputs, var_name, mc_fwd, stochastic=True, summary=False):
    if stochastic:
        logging.info('%s = sigmoid(Logit) and Logit ~ Normal(loc, scale)', var_name)
    with tf.name_scope(var_name):
        outputs = normal_sample(
            shape=shape,
            inputs=inputs,
            var_name='Logit',
            mc_fwd=mc_fwd,
            stochastic=stochastic,
            summary=summary
        )
    outputs = tf.nn.sigmoid(outputs)

    if summary:
        tf.summary.histogram(var_name, outputs)

    return outputs


def logistic_normal_sample(shape, inputs, var_name, mc_fwd, stochastic=True, summary=False):
    if stochastic:
        logging.info('%s = softmax(Logit) and Logit ~ Normal(loc, scale)', var_name)
    with tf.name_scope(var_name):
        outputs = normal_sample(
            shape=shape,
            inputs=inputs,
            var_name='Logit',
            mc_fwd=mc_fwd,
            stochastic=stochastic,
            summary=summary
        )
    outputs = tf.nn.softmax(outputs)

    if summary:
        tf.summary.histogram(var_name, outputs)

    return outputs


def multivariate_normal_sample(shape, inputs, var_name, mc_fwd, stochastic=True, summary=False):
    dim = shape[-1]
    if dim == 1:
        normal_sample(
            shape=shape,
            inputs=inputs,
            var_name=var_name,
            mc_fwd=mc_fwd,
            stochastic=stochastic,
            summary=summary
        )

    if not stochastic:
        return delta_tensor(
            shape=shape,
            inputs=inputs,
            var_name=var_name,
            activation=tf.nn.softplus,
            summary=summary
        )

    logging.info('%s ~ MultivariateNormal(loc, scale)', var_name)

    with tf.variable_scope(var_name):
        loc = delta_tensor(
            shape=shape,
            inputs=inputs,
            var_name='loc',
            activation=tf.identity,
            summary=summary
        )
        scale = delta_cholesky(
            shape=shape,
            inputs=inputs,
            var_name='scale',
            summary=summary
        )
        N = tf.contrib.distributions.MultivariateNormalTriL(loc=loc, scale_tril=scale)
        outputs = tf.cond(
            mc_fwd,
            true_fn=lambda: N.sample(),
            false_fn=lambda: loc
        )
    if summary:
        tf.summary.histogram(var_name, outputs)
    return outputs
