"""
:Authors: - Wilker Aziz
"""
import tensorflow as tf


def kl_from_q_to_standard_normal(mean, log_var):
    """
    Kullback-Leibler divergence KL(q||p) for q(z) = N(mu, std) and p(z) = N(0,I)

    :param mean: [B * M, dz]
    :param log_var: [B * M, dz]
    :return: [B * M]
    """
    # [B * M]
    return -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=-1)


def kl_diagonal_gaussians(mean1, log_var1, mean2, log_var2):
    """
    KL between q and p where
        q is N(mean1, var1)
        p is N(mean2, var2)

    References:
        - https://tgmstat.wordpress.com/2013/07/10/kullback-leibler-divergence/
        - https://en.wikipedia.org/wiki/Kullback–Leibler_divergence#Kullback.E2.80.93Leibler_divergence_for_multivariate_normal_distributions
        - https://arxiv.org/pdf/1611.01437.pdf

    :param mean1: [B * M, dz]
    :param var1: [B * M, dz]
    :param mean2: [B * M, dz]
    :param var2: [B * M, dz]
    :return: KL [B * M]
    """
    var1 = tf.exp(log_var1)
    var2 = tf.exp(log_var2)
    return 0.5 * tf.reduce_sum(log_var2 - log_var1 + (var1 + tf.square(mean1 - mean2)) / var2 - 1, axis=-1)