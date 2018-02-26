import tensorflow as tf
import logging
from dgm4nlp.tf.ssoftmax import botev_sampled_softmax_layer
from dgm4nlp.tf.ssoftmax import jean_sampled_softmax_layer
from dgm4nlp.tf.ssoftmax import botev_batch_sampled_softmax_layer


def logit_layer_for_text(
        nb_classes,  # V
        inputs,  # [B, T, dim]
        labels,  # [B, T]
        dim,
        nb_softmax_samples,  # S
        is_training,
        approximation='botev-batch',
        support=None,  # [S]
        importance=None,  # [S]
        name='logit',
):
    """
    Logit strategies for monolingual sequences.
    
    :param nb_classes: number of classes over which we define a softmax
    :param inputs: forward activations [B, T, dim]
    :param labels: target labels [B, T]
    :param dim: number of activations dim
    :param nb_softmax_samples: use between 0 and nb_classes to get an approximation
    :param is_training: for sampled approximations this switches between truncated/complete supports at training/prediction
    :param approximation: which approximation to use
        - 'botev': CSS with a shared support for all elements in a sequence
        - 'jean': a form of IS with shared negative support
        - 'botev-batch': CSS with a shared support for all sequences in batch
    :param support: a batch-wise shared support of probable and negative classes
        - necessary for botev-batch, ignored by others
    :param importance: importance of elements in support
        - necessary for botev-batch, ignored by others
    :return: logits [B * T, V|S] and targets [B * T]
    """
    batch_size = tf.shape(inputs)[0]
    longest = tf.shape(inputs)[1]
    if 0 < nb_softmax_samples < nb_classes:  # Here we employ a sampled softmax architecture
        logging.info('%s sampled-softmax=%s', name, approximation)
        if approximation == 'botev':  # Here we use CSS (Botev et al, 2017)
            with tf.variable_scope('botev'):
                # logits: [B, T, Vx|S]
                # targets: [B, T]
                logits, targets = botev_sampled_softmax_layer(
                    nb_classes=nb_classes,
                    nb_samples=nb_softmax_samples,
                    dim=dim,
                    labels=labels,  # [B, T]
                    inputs=inputs,  # [B, T, dim]
                    is_training=is_training
                )

                # For compatibility with the rest of the code
                # [B * T, V|S]
                logits = tf.reshape(logits, [batch_size * longest, -1])
                # [B * T]
                targets = tf.reshape(targets, [-1])
        elif approximation == 'botev-batch':
            if support is None or importance is None:
                raise ValueError('Softmax approximation "botev-batch" requires "support" and "importance"')
            with tf.variable_scope('botev-batch'):
                # logits: [B, T, V|S]
                # targets: [B, T]
                logits, targets = botev_batch_sampled_softmax_layer(
                    nb_classes=nb_classes,  # V
                    dim=dim,
                    labels=labels,  # [B, T]
                    support=support,  # [S]
                    importance=importance,  # [S]
                    inputs=inputs,  # [B, T, dim]
                    is_training=is_training
                )

                # For compatibility with the rest of the code
                # [B * M, Vy|S]
                logits = tf.reshape(logits, [batch_size * longest, -1])
                # [B * T]
                targets = tf.reshape(targets, [-1])
        elif approximation == 'jean':  # Here we use the method of Jean et al (2015) with uniform sampling
            with tf.variable_scope('jean'):
                # logits: [B * T, V|S]
                # targets: [B * T]
                logits, targets = jean_sampled_softmax_layer(
                    nb_classes=nb_classes,
                    nb_samples=nb_softmax_samples,
                    dim=dim,
                    labels=tf.reshape(labels, [batch_size * longest, 1]),  # [B * T, 1]
                    inputs=tf.reshape(inputs, [batch_size * longest, -1]),  # [B * M, dim]
                    is_training=is_training
                )
        else:
            raise ValueError('Unknown softmax approximation for text: %s' % approximation)

    else:  # Here we employ an exact softmax architecture
        # Here we compute logits
        # [B * T, V]
        logits = tf.contrib.layers.fully_connected(
            tf.reshape(inputs, [batch_size * longest, dim]),  # [B * T, dim]
            num_outputs=nb_classes,
            activation_fn=None
        )

        # Define targets
        # [B * T]
        targets = tf.reshape(labels, [-1])

    return logits, targets




def logit_layer_for_bitext(
        nb_classes,  # V
        inputs,  # [B, M, dim]
        outputs,  # [B, N]
        dim,
        nb_softmax_samples,  # S
        is_training,
        approximation='botev-batch',
        support=None,  # [S]
        importance=None,  # [S]
        name='logit'
):
    """
    Logit strategies for sequences where the inputs and the outputs are defined over parallel sequences.

    :param nb_classes: number of classes over which we define a softmax
    :param inputs: forward activations [B, M, dim]
    :param outputs: output labels [B, N]
    :param dim: number of activations dim
    :param nb_softmax_samples: use between 0 and nb_classes to get an approximation
    :param is_training: for sampled approximations this switches between truncated/complete supports at training/prediction
    :param approximation: which approximation to use
        - 'botev': CSS with a shared support for all elements in a sequence
        - 'botev-batch': CSS with a shared support for all sequences in batch
    :param support: a batch-wise shared support of probable and negative classes
        - necessary for botev-batch, ignored by others
    :param importance: importance of elements in support
        - necessary for botev-batch, ignored by others
    :return: logits [B * T, V|S] and targets [B * T]
    """

    batch_size = tf.shape(inputs)[0]      # B
    longest_input = tf.shape(inputs)[1]   # M
    longest_output = tf.shape(outputs)[1]  # N

    if 0 < nb_softmax_samples < nb_classes:  # Here we employ a sampled softmax architecture
        logging.info('%s sampled-softmax=%s', name, approximation)
        if approximation == 'botev':
            with tf.variable_scope('botev'):
                # logits: [B, M, V|S]
                # targets: [B, N]
                logits, targets = botev_sampled_softmax_layer(
                    nb_classes=nb_classes,
                    nb_samples=nb_softmax_samples,
                    dim=dim,
                    labels=outputs,  # [B, N]
                    inputs=inputs,  # [B, M, dim]
                    is_training=is_training
                )

                # For compatibility with the rest of the code
                # [B * M, V|S]
                logits = tf.reshape(logits, [batch_size * longest_input, -1])
                # [B * N]
                targets = tf.reshape(targets, [batch_size * longest_output])
        elif approximation == 'botev-batch':
            if support is None or importance is None:
                raise ValueError('Softmax approximation "botev-batch" requires "support" and "importance"')
            with tf.variable_scope('botev-batch'):
                # logits: [B, M, V|S]
                # targets: [B, N]
                logits, targets = botev_batch_sampled_softmax_layer(
                    nb_classes=nb_classes,  # V
                    dim=dim,
                    labels=outputs,  # [B, N]
                    support=support,  # [S]
                    importance=importance,  # [S]
                    inputs=inputs,  # [B, M, dim]
                    is_training=is_training
                )

                # For compatibility with the rest of the code
                # [B * M, V|S]
                logits = tf.reshape(logits, [batch_size * longest_input, -1])
                # [B * N]
                targets = tf.reshape(targets, [batch_size * longest_output])
        else:
            raise ValueError('Unknown softmax approximation for bitext: %s' % approximation)
    else:  # Here we employ an exact softmax architecture
        # [B * M, V]
        logits = tf.contrib.layers.fully_connected(
            tf.reshape(inputs, [batch_size * longest_input, dim]),  # [B * M, dim]
            num_outputs=nb_classes,
            activation_fn=None  # for logits
        )

        # Define targets
        # [B * N]
        targets = tf.reshape(outputs, [-1])

    # [B * M, V|S], [B * N]
    return logits, targets


