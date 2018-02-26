"""
:Authors: - Wilker Aziz
"""
import tensorflow as tf


def self_attention_layer(
        inputs,
        num_steps,
        units,
        activation=tf.nn.softmax,
        mask_diagonal=False,
        mask_value=float('-inf'),
        name='SelfAttention',
        reuse=None):
    """
    Compute self attention levels (masking invalid positions).

    :param inputs: [batch_size, max_time, dim]
    :param num_steps: number of steps per training instance [batch_size]
    :param units: number of query/key units
    :param activation: defaults to tf.nn.softmax for normalised attention
    :param mask_diagonal: defaults to False
    :param mask_value: defaults to -inf
    :param name: defaults to SelfAttention
    :param reuse: passed to tf layers (defaults to None)
    :return: [batch_size, max_time, max_time]
    """
    batch_size = tf.shape(inputs)[0]  # B
    longest = tf.shape(inputs)[1]  # M
    with tf.variable_scope(name):
        # [B, M, d]
        queries = tf.layers.dense(inputs, units=units, name='queries', reuse=reuse)
        keys = tf.layers.dense(inputs, units=units, name='keys', reuse=reuse)
        # [B, M, M]
        scores = tf.matmul(
            queries,  # [B, M, d]
            keys,  # [B, M, d]
            transpose_b=True
        )
        # mask invalid logits
        scores = tf.where(
            # make the boolean mask [B, M, M]
            condition=tf.tile(
                # make the boolean mask [B, 1, M]
                tf.expand_dims(
                    # get a boolean mask [B, M]
                    tf.sequence_mask(num_steps, maxlen=longest),
                    1
                ),
                [1, longest, 1]
            ),
            x=scores,
            y=tf.ones(shape=[batch_size, longest, longest]) * mask_value
        )
        # mask diagonal
        if mask_diagonal:
            scores += tf.diag(tf.fill([tf.shape(scores)[-1]], mask_value))
        # Normalise attention
        # [B, M, M]
        return activation(scores)

