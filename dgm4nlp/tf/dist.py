

"""



For KL, see [Table 3](http://www.mast.queensu.ca/~linder/pdf/GiAlLi13.pdf)

:Authors: - Wilker Aziz
"""
import tensorflow as tf
import numpy as np


def tf_beta_fn(a, b):
    # A useful identity:
    #   B(a,b) = exp(log Gamma(a) + log Gamma(b) - log Gamma(a+b))
    # but here we simply exponentiate tf.lbeta instead, feel free to use whichever version you prefer
    return tf.exp(tf.lbeta(tf.concat([tf.expand_dims(a, -1), tf.expand_dims(b, -1)], -1)))


class ParameterSpec:
    """
    This class helps predict parameters by setting an appropriate activation_fn
    """

    def __init__(self, activation_fn, name: str):
        self.activation_fn = activation_fn
        self.name = name


class Location(ParameterSpec):

    def __init__(self, name: str):
        super(Location, self).__init__(tf.identity, name)


class Scale(ParameterSpec):

    def __init__(self, name: str):
        super(Scale, self).__init__(tf.nn.softplus, name)


class Rate(ParameterSpec):

    def __init__(self, name: str):
        super(Rate, self).__init__(tf.nn.softplus, name)


class Shape(ParameterSpec):

    def __init__(self, name: str):
        super(Shape, self).__init__(tf.nn.softplus, name)


class Probability(ParameterSpec):

    def __init__(self, name: str):
        super(Probability, self).__init__(tf.sigmoid, name)


class Distribution:

    def __init__(self, param_specs: 'list[ParameterSpec]', image: str):
        self._param_specs = param_specs
        self._image = image

    def image(self) -> str:
        return self._image

    def num_params(self) -> int:
        return len(self._param_specs)

    def param_specs(self) -> 'list[ParameterSpec]':
        return self._param_specs

    def param_spec(self, param: int) -> ParameterSpec:
        return self._param_specs[param]

    def activation_fn(self, param: int):
        return self._param_specs[param].activation_fn

    def mean(self, params: list):
        pass

    def random_standard(self, shape, dtype=tf.float32, seed=None, name=None):
        pass

    def random(self, shape, params: list, dtype=tf.float32, seed=None, name=None):
        pass

    def kl(self, params_i: list, params_j: list):
        pass

    def kl_from_standard(self, params: list):
        pass


class Bernoulli(Distribution):

    def __init__(self, negative=0, positive=1, standard=[0.5]):
        super(Bernoulli, self).__init__(param_specs=[Probability('p')], image='prob')
        self._negative = negative
        self._positive = positive
        self._standard = standard

    def mean(self, params: list):
        return params[0]

    def random_standard(self, shape, dtype=tf.float32, seed=None, name=None):
        return self.random(shape, params=self._standard, dtype=dtype, seed=seed, name=name)

    def random(self, shape, params: list, dtype=tf.float32, seed=None, name=None):
        """
        X ~ Bernoulli(\theta)

        :param shape:
        :param params: [p] probability of the positive class
        :param dtype:
        :param seed:
        :param name:
        :return:
        """
        p = params[0]
        return tf.where(
            tf.less(
                tf.random_uniform(shape, minval=0., maxval=1., dtype=tf.float32, seed=seed, name=name),
                p),
            x=tf.cast(self._positive, dtype),
            y=tf.cast(self._negative, dtype))

    def kl(self, params_i: list, params_j: list):
        p_i = params_i[0]
        q_i = 1 - p_i
        p_j = params_j[0]
        q_j = 1 - p_j
        return p_i * (tf.log(p_i) - tf.log(p_j)) + q_i * (tf.log(q_i) - tf.log(q_j))

    def kl_from_standard(self, params: list):
        return self.kl(params_i=params, params_j=self._standard)


class Exponential(Distribution):

    def __init__(self):
        super(Exponential, self).__init__(param_specs=[Rate('lambda')], image='positive')

    def mean(self, params: list):
        rate = params[0]
        return 1. / rate

    def random_standard(self, shape, dtype=tf.float32, seed=None, name=None):
        u = tf.random_uniform(shape, minval=0, maxval=1, dtype=dtype, seed=seed, name=name)
        return -tf.log(u)

    def random(self, shape, params: list, dtype=tf.float32, seed=None, name=None):
        rate = params[0]
        return self.random_standard(shape, dtype=dtype, seed=seed, name=name) / rate

    def kl(self, params_i: list, params_j: list):
        rate_i = params_i[0]
        rate_j = params_j[0]
        return tf.log(rate_i) - tf.log(rate_j) + rate_j / rate_i - 1

    def kl_from_standard(self, params: list):
        rate = params[0]
        return tf.log(rate) + 1 / rate - 1


class ApproxPoisson(Distribution):

    def __init__(self, standard: list):
        super(ApproxPoisson, self).__init__(param_specs=[Rate('lambda')], image='positive')
        self._standard = standard

    def mean(self, params: list):
        rate = params[0]
        return 1 / rate

    def random_standard(self, shape, dtype=tf.float32, seed=None, name=None):
        return self.random(shape, params=self._standard, dtype=dtype, seed=seed, name=name)

    def random(self, shape, params: list, dtype=tf.float32, seed=None, name=None):
        rate = params[0]
        eps = tf.random_normal(shape, mean=0., stddev=1., dtype=dtype, seed=seed, name=name)
        return tf.square(tf.sqrt(rate) + 0.5 * eps)

    def kl(self, params_i: list, params_j: list):
        rate_i = params_i[0]
        rate_j = params_j[0]
        return self._normal.kl([tf.sqrt(rate_i), 0.5], [tf.sqrt(rate_j), 0.5])

    def kl_from_standard(self, params: list):
        rate = params[0]
        return self._normal.kl([tf.sqrt(rate), 0.5], [tf.sqrt(self._standard[0]), 0.5])


class Kuma(Distribution):

    def __init__(self, standard: list, num_terms=10):
        super(Kuma, self).__init__(param_specs=[Shape('a'), Shape('b')], image='prob')
        self._standard = standard
        self._num_terms = num_terms

    def raw_moment(self, params: list, n: int):
        a, b = params
        return tf.exp(tf.log(b) + tf.lgamma(1 + n/a) + tf.lgamma(b) - tf.lgamma(1 + b + n/a))

    def mean(self, params: list):
        return self.raw_moment(params, 1)  # mean = 1st raw moment

    def random_standard(self, shape, dtype=tf.float32, seed=None, name=None):
        return self.random(shape, params=self._standard, dtype=dtype, seed=seed, name=name)

    def random(self, shape, params: list, dtype=tf.float32, seed=None, name=None):
        a, b = params
        u = tf.random_uniform(shape, minval=0., maxval=1., dtype=dtype, seed=seed, name=name)
        return (1. - (1. - u) ** (1. / b)) ** (1. / a)

    def kl(self, params_i: list, params_j: list):
        """
        KL(Kuma(a', b') || Beta(a, b))

        :param params_i: [kuma_a, kuma_b]
        :param params_j: [beta_a, beta_b]
        :return:
        """
        kuma_a, kuma_b = params_i
        beta_a, beta_b = params_j
        term1 = (kuma_a - beta_a) / kuma_a * (- np.euler_gamma - tf.digamma(kuma_b) - 1.0 / kuma_b)
        term1 += tf.log(kuma_a * kuma_b) + tf.lbeta(
            tf.concat([tf.expand_dims(beta_a, -1), tf.expand_dims(beta_b, -1)], -1))
        term1 += - (kuma_b - 1) / kuma_b
        # Truncated Taylor expansion around 1
        taylor = tf.zeros(tf.shape(kuma_a))
        for m in range(1, self._num_terms + 1):  # m should start from 1 (otherwise betafn will be inf)!
            taylor += tf_beta_fn(m / kuma_a, kuma_b) / (m + kuma_a * kuma_b)
        term2 = (beta_b - 1) * kuma_b * taylor
        return term1 + term2  # tf.maximum(0., term1 + term2)

    def kl_from_standard(self, params: list):
        return self.kl(params_i=params, params_j=self._standard)


class Kuma1(Distribution):

    def __init__(self, standard: list, num_terms=10):
        super(Kuma1, self).__init__(param_specs=[Probability('a')], image='prob')
        self._kuma2 = Kuma(standard=[standard[0], 1 - standard[0]], num_terms=num_terms)

    def raw_moment(self, params: list, n: int):
        a = params[0]
        b = 1 - a
        return self._kuma2.raw_moment([a, b], n)

    def mean(self, params: list):
        a = params[0]
        b = 1 - a
        return self._kuma2.mean([a, b])

    def random_standard(self, shape, dtype=tf.float32, seed=None, name=None):
        return self._kuma2.random_standard(shape, dtype=dtype, seed=seed, name=name)

    def random(self, shape, params: list, dtype=tf.float32, seed=None, name=None):
        a = params[0]
        b = 1 - a
        return self._kuma2.random(shape, params=[a, b], dtype=dtype, seed=seed, name=name)

    def kl(self, params_i: list, params_j: list):
        """
        KL(Kuma(a', b') || Beta(a, b))

        :param params_i: [kuma_a, kuma_b]
        :param params_j: [beta_a, beta_b]
        :return:
        """
        a_i = params_i[0]
        b_i = 1 - a_i
        a_j = params_j[0]
        b_j = 1 - a_j
        return self._kuma2.kl([a_i, b_i], [a_j, b_j])

    def kl_from_standard(self, params: list):
        a = params[0]
        b = 1 - a
        return self._kuma2.kl_from_standard([a, b])


class LocationScale(Distribution):
    """
    E ~ Dist(0, 1)
    location + scale * E ~ Dist(location, scale)
    """

    def __init__(self):
        super(LocationScale, self).__init__(param_specs=[Location('location'), Scale('scale')], image='real')

    def random(self, shape, params: list, dtype=tf.float32, seed=None, name=None):
        """
        Representation:

            E ~ Dist(0, 1)
            l + s * E ~ Dist(l, s)

        :param shape:
        :param params: [location, scale]
        :param dtype:
        :param seed:
        :param name:
        :return:
        """
        location, scale = params
        epsilon = self.random_standard(shape, dtype=dtype, seed=seed, name=name)
        return location + scale * epsilon


class Normal(LocationScale):

    def __init__(self):
        super(Normal, self).__init__()

    def mean(self, params: list):
        return params[0]  # [mean=location, scale]

    def random_standard(self, shape, dtype=tf.float32, seed=None, name=None):
        """
        X ~ N(0, I).

        :param shape:
        :param dtype:
        :param seed:
        :param name:
        :return:
        """
        return tf.random_normal(shape, mean=0., stddev=1., dtype=dtype, seed=seed, name=name)

    def kl(self, params_i: list, params_j: list):
        location_i, scale_i = params_i  # [mean, std]
        location_j, scale_j = params_j  # [mean, std]
        var_i = scale_i ** 2
        var_j = scale_j ** 2
        term1 = 1 / (2 * var_j) * ((location_i - location_j) ** 2 + var_i - var_j)
        term2 = tf.log(scale_j) - tf.log(scale_i)
        return term1 + term2  # tf.reduce_sum(term1 + term2, axis=-1)

    def kl_from_standard(self, params: list):
        """
        KL( N(\mu, \sigma^2) || N(0,I) )

        :param params: [location, scale] each with shape [B, dz]
        :return: [B]
        """
        location, scale = params
        #return -0.5 * tf.reduce_sum(1 + 2 * tf.log(scale) - tf.square(location) - tf.square(scale), axis=-1)
        return -0.5 * (1 + 2 * tf.log(scale) - tf.square(location) - tf.square(scale))


class Laplace(LocationScale):

    def __init__(self, representation='normal'):
        """
        :param representation: choice of representation of the Laplace variable
            - exponential:
                * |X| ~  Exp(I)
                * Y ~ Bernoulli(0.5)
                * Y |X| ~ L(0,I)
            - normal (default):
                * X_1, X_2, X_3, X_4 ~ N(0, I)
                * X_1 * X_2 - X_3 * X_4 ~ L(0, I)
        """
        super(Laplace, self).__init__()
        self._representation = representation
        self._bernoulli = Bernoulli(negative=-1., positive=1.)

    def mean(self, params: list):
        return params[0]  # [mean=location, scale]

    def random_standard(self, shape, dtype=tf.float32, seed=None, name=None):
        if self._representation == 'exponential':
            # Representation: X ~ L(0, I)
            #  |X| ~  Exp(I)
            #  Y ~ Bernoulli(0.5)
            #  Y |X| ~ L(0,I)
            abs_x = tf.random_gamma(shape, alpha=0., beta=1., dtype=dtype, seed=seed, name=name)
            sign = self._bernoulli.random(shape, [0.5], dtype=dtype, seed=seed, name=name)
            return sign * abs_x
        else:  # defaults to 'normal'
            # Representation: X ~ L(0, I) then
            #  X1, X2, X3, X4 ~ N(0, I) => X1 * X2 - X3 * X4 ~ L(0, I)
            x1 = tf.random_normal(shape, dtype=dtype, seed=seed, name=name)
            x2 = tf.random_normal(shape, dtype=dtype, seed=seed, name=name)
            x3 = tf.random_normal(shape, dtype=dtype, seed=seed, name=name)
            x4 = tf.random_normal(shape, dtype=dtype, seed=seed, name=name)
            return x1 * x2 - x3 * x4

    def kl(self, params_i: list, params_j: list):
        location_i, scale_i = params_i
        location_j, scale_j = params_j
        abs_diff = tf.abs(location_i - location_j)
        term1 = tf.log(scale_j) - tf.log(scale_i)
        term2 = abs_diff / scale_j
        term3 = scale_i / scale_j * tf.exp(- abs_diff / scale_i)
        return term1 + term2 + term3 - 1

    def kl_from_standard(self, params: list):
        """
        KL( L(\mu, \beta) || L(0, I) )


        :param params: [location, scale] each with shape [B * M, dz]
        :return: [B * M]
        """
        location, scale = params
        # [B * M]
        #return tf.reduce_sum(-tf.log(scale) + tf.abs(location) + scale * tf.exp(- tf.abs(location) / scale) - 1,
        #                     axis=-1)
        return -tf.log(scale) + tf.abs(location) + scale * tf.exp(- tf.abs(location) / scale) - 1


class Gumbel(LocationScale):

    def __init__(self):
        super(Gumbel, self).__init__()

    def mean(self, params: list):
        location, scale = params
        return location + scale * np.euler_gamma

    def random_standard(self, shape, dtype=tf.float32, seed=None, name=None):
        u = tf.random_uniform(shape, minval=0, maxval=1., dtype=dtype, seed=seed, name=name)
        return -tf.log(-tf.log(u))

    def kl(self, params_i: list, params_j: list):
        raise ValueError('Missing an expression for KL')
        location_i, scale_i = params_i
        location_j, scale_j = params_j
        scale_ratio_ij = scale_i / scale_j
        term1 = tf.log(scale_j) - tf.log(scale_i)
        term2 = np.euler_gamma * (scale_ratio_ij - 1)
        log_term3 = (location_j - location_i) / scale_j + tf.lgamma(scale_ratio_ij + 1)
        return term1 + term2 + tf.exp(log_term3) - 1

    def kl_from_standard(self, params: list):
        raise ValueError('Missing an expression for KL')
        location, scale = params
        return - tf.log(scale) + np.euler_gamma * (scale - 1) + tf.exp(-location + tf.lgamma(scale + 1)) - 1


class Gumbel1(Distribution):

    def __init__(self):
        super(Gumbel1, self).__init__(param_specs=[Location('location')], image='real')

    def mean(self, params: list):
        location = params[0]
        return location + np.euler_gamma

    def random_standard(self, shape, dtype=tf.float32, seed=None, name=None):
        u = tf.random_uniform(shape, minval=0, maxval=1., dtype=dtype, seed=seed, name=name)
        return -tf.log(-tf.log(u))

    def random(self, shape, params: list, dtype=tf.float32, seed=None, name=None):
        location = params[0]
        u = self.random_standard(shape, dtype=dtype, seed=seed, name=name)
        return location + u

    def kl(self, params_i: list, params_j: list):
        raise ValueError('Missing an expression for KL')

    def kl_from_standard(self, params: list):
        raise ValueError('Missing an expression for KL')


def two_params_cls(name) -> Distribution:
    if name == 'normal' or name == 'gaussian':
        return Normal
    elif name == 'laplace':
        return Laplace
    elif name == 'gumbel':
        return Gumbel
    elif name == 'kuma':
        return Kuma
    else:
        raise ValueError('Unknown distribution: %s' % name)


def location_scale_cls(name) -> LocationScale:
    if name == 'normal' or name == 'gaussian':
        return Normal
    elif name == 'laplace':
        return Laplace
    elif name == 'gumbel':
        return Gumbel
    else:
        raise ValueError('Unknown distribution: %s' % name)