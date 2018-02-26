"""
:Authors: - Wilker Aziz
"""
import tensorflow as tf


def simple_gcn_layer(
        inputs,
        adjacencies,
        units,
        activation=tf.nn.relu,
        loop=True,
        reverse=True,
        residual=True,
        name='SimpleGCN',
        reuse=None):
    """
    A simple GCN layer.

    :param inputs: [batch_size, max_step, dim]
    :param adjacencies: [batch_size, max_step, max_step]
    :param units: number of GCN units (dim)
    :param activation: defaults to tf.nn.relu
    :param loop: independent parameters for self loop
    :param reverse: independent parameters for reversed edges
    :param name: defaults to 'SimpleGCN'
    :param reuse: passed to tf layers (defaults to None)
    :return: [batch_size, max_step, units]
    """
    # TODO: add gates
    with tf.variable_scope(name):
        # [B, M, d]
        tensors = []
        ho = tf.layers.dense(inputs, units=units, activation=tf.identity,name='out', reuse=reuse)
        ho = tf.matmul(
            adjacencies,  # [B, M, M]
            ho  # [B, M, d]
        )
        tensors.append(ho)
        if reverse:
            hi = tf.layers.dense(inputs, units=units, activation=tf.identity, name='in', reuse=reuse)
            hi = tf.matmul(
                adjacencies,  # [B, M, M]
                hi,  # [B, M, d]
                transpose_a=True
            )
            tensors.append(hi)
        if loop:
            hs = tf.layers.dense(inputs, units=units, activation=tf.identity,name='self', reuse=reuse)
            tensors.append(hs)
    outputs = activation(tf.add_n(tensors))
    if residual:
        outputs += inputs
    return outputs
