"""
:Authors: - Wilker Aziz
"""
import tensorflow as tf
from dgm4nlp.tf.utils import dense


def self_attention_layer(
        inputs,
        num_steps,
        units,
        dropout=None,
        is_training=None,
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
        queries = dense(inputs, units=units, dropout=dropout, is_training=is_training, name='queries', reuse=reuse)
        keys = dense(inputs, units=units, dropout=dropout, is_training=is_training, name='keys', reuse=reuse)
        # [B, M, M]
        scores = tf.matmul(
            queries,  # [B, M, d]
            keys,  # [B, M, d]
            transpose_b=True
        )
        # mask invalid logits
        # [B, M, M]
        condition = tf.tile(
            # make the boolean mask [B, 1, M]
            tf.expand_dims(
                # get a boolean mask [B, M]
                tf.sequence_mask(num_steps, maxlen=longest),
                1
            ),
            [1, longest, 1]
        )
        scores = tf.where(
            # make the boolean mask [B, M, M]
            condition=condition,
            x=scores,
            y=tf.ones(shape=[batch_size, longest, longest]) * mask_value
        )
        # mask diagonal
        if mask_diagonal:
            scores += tf.diag(tf.fill([tf.shape(scores)[-1]], mask_value))
        # Normalise attention
        # [B, M, M]
        #outputs = tf.where(
        #    condition=condition,
        #    x=activation(scores),
        #    y=tf.zeros_like(scores)
        #)
        return activation(scores)


def attention_logits(
        inputs,   # [B, M, d]
        outputs,  # [B, N, d]
        units,
        dropout=None,
        is_training=None,
        name='attention-layer',
        reuse=None
):
    with tf.variable_scope(name, reuse=reuse):
        # [B, M, d]
        keys = dense(
            inputs=inputs,  # [B, M, dx]
            units=units,
            activation=None,
            name='keys',
            dropout=dropout,
            is_training=is_training
        )
        # [B, N, d]
        queries = dense(
            inputs=outputs,  # [B, N, dy]
            units=units,
            activation=None,
            name='queries',
            dropout=dropout,
            is_training=is_training
        )

        # prediction of Categorical parameters of P(A_j|x_1^m, y_<j) via dot product
        # [B, N, M]
        logits = tf.matmul(
            queries,  # [B, N, dh]
            keys,  # [B, M, dh]
            transpose_b=True
        )

    # [B, N, M]
    return logits
