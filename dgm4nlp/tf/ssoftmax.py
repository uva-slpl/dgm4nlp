"""
This module provides two sampled softmax implementations.

1. The first one reproduces [Jean et al, 2015](https://arxiv.org/pdf/1412.2007.pdf)

    I copied this code from tf [r1.2](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/nn_impl.py)
    because I needed access to _compute_sampled_logits which is used inside tf.nn.sampled_softmax_loss.

2. The second one is a version of [Botev et al, 2017](http://proceedings.mlr.press/v54/botev17a/botev17a.pdf)


The APIs of the methods above differ as they are not intended to adress the very same scenarios.
Use method 1 if you simply want a sampled softmax to be directly plugged into a loss function
(but where additionally you have a reason to use the normalised probability as well).
Use method 2 if you want a fast softmax approximation (though slower than method 1) which is not meant to be
directly plugged into a loss, but still provides a consistent use of labels as to make a loss term via marginalisation
for example.

Consider the scenario where we need to compute a distribution P(Y|X=x) = Cat(softmax(f(x))) where x is some
 given conditioning context. If the support of the distribution is to large, the softmax becomes a serious bottleneck.
 If we mean to compute a loss such as -\log P(Y=y|X=x) for some observation (x, y), then tf offers a lot of sampled
 softmax approximations, where we compute -\log P(Y=y|X=x, S=s) for some sampled subset s of the entire support
 (which includes the true class, i.e., x \in s).

Method 1 below (sampled_softmax_layer) is useful when, besides computing a loss, we have some use for the truncated
 distribution itself, that is, P(Y|X=x, S=s). The layer takes batches without time dimensions (thus we have to collapse
 sample and time dimensions before using the layer). This layer wraps tf code which was not exposed
 in the public API (namely, _compute_sampled_logits) with small modifications as well as a training/prediction logic.
 At training time we use the sampled support s, at prediction (validation/test) time we use the complete support, thus
 producing P(Y|X=x) without approximations. Here the support s is made of the target class plus a set of negative
 classes. The negative classes are shared across all instances in a batch and it does not contain any of the classes
 observed in the batch (as those are positive for some training instance).

Method 2 below (sampled_softmax_css_layer) is useful when the approximate distribution is not a term in the loss,
 but does contribute to it indirectly (for example, through marginalisation of some latent variable). Consider the case
 where our distribution whose support is too large results from marginalisation of some latent variable a.
 Thus, P(Y=y|X=x) = \sum_a P(A=a|X=x) P(Y=y|X=x, A=a).
 The loss term would be - \log P(Y=y|X=x) for some observation (x, y), but to compute it efficiently, we need to
 tractably approximate P(Y|X=x,A=a).
 For that, we have the second method. Moreover, we taylored the implementation for sequences, thus it can take a
  batch of input sequences and produce batches of distributions over output sequences, where these sequences may
  differ in length. Crucially, the support s is sampled for each sequence (not for each word in the sequence).
 For example, if our ith pair of observations is (x_1^m, y_1^n) then the support is specific to this instance,
 i.e. s = s^(i), and contains y_1^n (without repetitions) and a set of negative classes (no repetitions).
 Note that this is really crucial, in method 1, the support s would differ for each conditioning word which would make
 it hard to approximate the marginalisation.


:Authors: - Wilker Aziz
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
import numpy as np


def _sum_rows(x):
  """Returns a vector summing up each row of the matrix x."""
  # _sum_rows(x) is equivalent to math_ops.reduce_sum(x, 1) when x is
  # a matrix.  The gradient of _sum_rows(x) is more efficient than
  # reduce_sum(x, 1)'s gradient in today's implementation. Therefore,
  # we use _sum_rows(x) in the nce_loss() computation since the loss
  # is mostly used for training.
  cols = array_ops.shape(x)[1]
  ones_shape = array_ops.stack([cols, 1])
  ones = array_ops.ones(ones_shape, x.dtype)
  return array_ops.reshape(math_ops.matmul(x, ones), [-1])


def compute_sampled_logits_css(
        weights,  # [V, d]
        biases,   # [V]
        probable,   # [B, P]
        inputs,   # [B, T, d]
        num_sampled,
        num_classes,
        minval=1,  # use this to control what's the first valid class
        bias_correction=True
):
    """

    :param weights: class embedding matrix
    :param biases: class biases
    :param probable: probable sequences [B, P]
    :param inputs: forward activations [B, T, d]
    :param num_sampled: number of random uniform samples
    :param num_classes: total number of classes
    :param minval: initial class (defaults to 1)
    :param bias_correction: whether or not to employ Barber's bias correction for negative classes (defaults to True)
    :return:
        logits: [B, T, S]
        support: [B, S]
        The support will include all of the probable classes and it will have no repetitions.
        Supports are sampled for each training sequence (and the entire sequence is considered probable).
    """
    batch_size = tf.shape(inputs)[0]
    num_probable = tf.shape(inputs)[1]
    # [B, N]
    uniform_samples = tf.random_uniform(
        [batch_size, num_sampled],
        minval=minval, maxval=num_classes, dtype=probable.dtype
    )
    # [B, P + N]
    all_samples = tf.concat([probable, uniform_samples], axis=-1)
    # Sort the samples in the support
    # [B, P + N]
    support, indices = tf.nn.top_k(all_samples, k=tf.shape(all_samples)[-1], sorted=True)
    # Extends the tensor to the left with a column whose elements are larger than those in the original first column
    # [B, 1 + P + N]
    left_augmented = tf.concat([tf.expand_dims(support[:, 0] + 1, -1), support], axis=-1)
    # Create a mask that sets duplicates to zero
    # [B, P + N]
    support_mask = tf.not_equal(left_augmented[:, 1:], left_augmented[:, :-1])
    # Cleans the support as to have non-zero labels appear only once
    # [B, S] where S = P + N
    support *= tf.cast(support_mask, support.dtype)
    # Extends the support with the zero class
    # support = tf.concat([support, tf.zeros([batch_size, 1], dtype=support.dtype)], axis=-1)

    # Embed classes in support
    # [B, S, d]
    c_emb = tf.nn.embedding_lookup(
        weights,  # [V, d]
        support   # [B, S]
    )

    # [B, T, S]
    logits = tf.matmul(
        inputs,  # [B, T, d]
        c_emb,   # [B, S, d]
        transpose_b=True
    )

    # Incorporate class bias
    # [B, S]
    c_bias = tf.nn.embedding_lookup(
        biases,  # [V]
        support  # [B, S]
    )
    # [B, T, S]
    logits += tf.expand_dims(c_bias, 1)  # this makes c_bias [B, 1, S]

    if bias_correction:  # as in (Botev et al, 2017)
        # Incorporate bias correction terms
        # Get the positions in the support where probable labels ended up
        # [B, S]
        probable_indices = tf.less(indices, num_probable)
        # Number of unique probable classes
        # [B]
        num_unique_probable = tf.reduce_sum(
            tf.cast(  # cast to float so I can sum
                # mask the probable indices for uniqueness
                tf.logical_and(probable_indices, support_mask),
                tf.float32
            ),
            -1
        )
        # Number of unique negative classes
        # [B]
        num_unique_negative = tf.reduce_sum(
            tf.cast(  # cast to float so I can sum
                # mask the negative indices for uniqueness
                tf.logical_and(
                    # an index that's not in the probable set is in the negative set
                    tf.logical_not(probable_indices),
                    support_mask
                ),
                tf.float32
            ),
            -1
        )
        # Correction term for negative samples
        #  as in section 4 of http: // web4.cs.ucl.ac.uk / staff / D.Barber / publications / AISTATS2017.pdf
        #    \kappa(c) = 1.0 / (|neg| * q(c))
        #  here we use a uniform proposal q(c) over all classes except those in the probable set
        #    q(c) = 1.0 / (V - |pos|)
        #  thus
        #    \kappa(c) = (V - |pos|) / (|neg|)
        # [B]
        log_kappa_neg = tf.log((num_classes - num_unique_probable)) - tf.log(num_unique_negative)
        # We make a kappa matrix for all label (but positive labels get kappa=1)
        # [B, S]
        log_kappa = tf.where(
            probable_indices,  # [B, S]
            tf.zeros(tf.shape(support), dtype=tf.float32),  # no penalties for probable elements
            tf.ones(tf.shape(support), dtype=tf.float32) * tf.expand_dims(log_kappa_neg, 1),  # penalty computed above
        )
        # [B, T, S]
        logits += tf.expand_dims(log_kappa, 1)  # expand log_kappa to have a time dimension

    # Sets logits of duplicate labels to -inf
    # [B, S]
    logits_mask = tf.log(tf.cast(support_mask, logits.dtype) + tf.contrib.keras.backend.epsilon())
    # [B, T, S]
    logits -= tf.expand_dims(logits_mask, 1)  # this makes the mask [B, 1, S]

    # [B, T, S], [B, S]
    return logits, support


def sampled_softmax_css_layer(
        nb_classes,
        nb_samples,
        dim,
        labels,  # [B, N]
        inputs,  # [B, M, d]
        is_training):
    """
    Creates a sampled softmax layer that can be used for marginalisation.

    :param nb_classes: total number of classes
    :param nb_samples: number of uniform samples
    :param dim: input (and class) dimensionality d
    :param labels: batch of gold sequences [B, N]
    :param inputs: forward activations [B, M, d]
        Here the sequence of activations do not need to have the same length as labels.
    :param is_training: in training we sample the support, in test/validation we use the entire support
    :return:
        logits: [B, M, S]
        targets: [B, N]

        The support has size S and is sampled for each training instance (a sequence).
         For the ith sequence, it includes all classes in labels[i] plus up to `nb_samples` random classes.
         The support does not contain repetitions of non-zero classes.
    """

    # Projection for classes
    # [V, d]
    c_embeddings = tf.get_variable(
        name='class_embedding',
        initializer=tf.random_uniform_initializer(),
        shape=[nb_classes, dim])
    # [V]
    c_biases = tf.get_variable(
        name='class_bias',
        initializer=tf.random_uniform_initializer(),
        shape=[nb_classes])

    def sampled_softmax():
        """
        In training we use a (fast) sampled softmax approximation.

        :return:
            logits: [B, M, S]
            targets: [B, N]
        """

        # logits: [B, M, S]
        # support: [B, S]
        logits, support = compute_sampled_logits_css(
            weights=c_embeddings,
            biases=c_biases,
            probable=labels,  # [B, N]
            inputs=inputs,  # [B, M, d]
            num_sampled=nb_samples,
            num_classes=nb_classes,
        )

        # Targets are the position in the support where the gold labels in the batch occurred
        # we make one-hot vectors where 1s signal the position where the gold label ended up in the support
        # we are guaranteed to find the gold label in the support because the gold label is part of the probable set
        # we are also guaranteed to find it only once, because the support does not have duplicate entries
        # [B, N, S]
        tgtpmf = tf.cast(  # cast to float to make a valid pmf
            tf.equal(  # compare whether a label in the support matches a gold label
                tf.expand_dims(labels, 2),  # make (gold) labels [B, N, 1]
                tf.expand_dims(support, 1)  # make labels in support [B, 1, S]
            ),
            tf.float32
        )

        # Targets
        # [B, N]
        targets = tf.argmax(tgtpmf, axis=-1)

        # [B, M, S], [B, N]
        return logits, targets

    def exact_softmax():
        """
        In validation/test we use the (slow) exact softmax.

        :return:
            logits: [B, M, V]
            targets: [B, N]
        """

        # Logits
        # [B * M, V]
        logits = tf.matmul(
            tf.reshape(inputs, [-1, dim]),  # [B * M, d]
            tf.transpose(c_embeddings)  # [d, V]
        )
        logits = tf.nn.bias_add(logits, c_biases)
        # [B, M, V]
        logits = tf.reshape(logits, [tf.shape(labels)[0], -1, nb_classes])

        # Targets
        # [B, N]
        targets = labels

        return logits, targets

    # depending on training/prediction phase we get approximate/exact logits and targets
    return tf.cond(is_training, true_fn=sampled_softmax, false_fn=exact_softmax)


def compute_sampled_logits(weights,
                            biases,
                            labels,
                            inputs,
                            num_sampled,
                            num_classes,
                            num_true=1,
                            sampled_values=None,
                            subtract_log_q=True,
                            remove_accidental_hits=False,
                            partition_strategy="mod",
                            name=None):
    """Helper function for nce_loss and sampled_softmax_loss functions.
    Computes sampled output training logits and labels suitable for implementing
    e.g. noise-contrastive estimation (see nce_loss) or sampled softmax (see
    sampled_softmax_loss).
    Note: In the case where num_true > 1, we assign to each target class
    the target probability 1 / num_true so that the target probabilities
    sum to 1 per-example.
    Args:
      weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
          objects whose concatenation along dimension 0 has shape
          `[num_classes, dim]`.  The (possibly-partitioned) class embeddings.
      biases: A `Tensor` of shape `[num_classes]`.  The (possibly-partitioned)
          class biases.
      labels: A `Tensor` of type `int64` and shape `[batch_size,
          num_true]`. The target classes.  Note that this format differs from
          the `labels` argument of `nn.softmax_cross_entropy_with_logits`.
      inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
          activations of the input network.
      num_sampled: An `int`.  The number of classes to randomly sample per batch.
      num_classes: An `int`. The number of possible classes.
      num_true: An `int`.  The number of target classes per training example.
      sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
          `sampled_expected_count`) returned by a `*_candidate_sampler` function.
          (if None, we default to `log_uniform_candidate_sampler`)
      subtract_log_q: A `bool`.  whether to subtract the log expected count of
          the labels in the sample to get the logits of the true labels.
          Default is True.  Turn off for Negative Sampling.
      remove_accidental_hits:  A `bool`.  whether to remove "accidental hits"
          where a sampled class equals one of the target classes.  Default is
          False.
      partition_strategy: A string specifying the partitioning strategy, relevant
          if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
          Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
      name: A name for the operation (optional).
    Returns:
      out_logits, out_labels: `Tensor` objects each with shape
          `[batch_size, num_true + num_sampled]`, for passing to either
          `nn.sigmoid_cross_entropy_with_logits` (NCE) or
          `nn.softmax_cross_entropy_with_logits` (sampled softmax).
    """

    if isinstance(weights, variables.PartitionedVariable):
        weights = list(weights)
    if not isinstance(weights, list):
        weights = [weights]

    with ops.name_scope(name, "compute_sampled_logits",
                        weights + [biases, inputs, labels]):
        if labels.dtype != dtypes.int64:
            labels = math_ops.cast(labels, dtypes.int64)
        labels_flat = array_ops.reshape(labels, [-1])

        # Sample the negative labels.
        #   sampled shape: [num_sampled] tensor
        #   true_expected_count shape = [batch_size, 1] tensor
        #   sampled_expected_count shape = [num_sampled] tensor
        if sampled_values is None:
            # TODO: make the sampler an option
            sampler = tf.nn.uniform_candidate_sampler
            # candidate_sampling_ops.log_uniform_candidate_sampler
            sampled_values = sampler(
                true_classes=labels,
                num_true=num_true,
                num_sampled=num_sampled,
                unique=True,
                range_max=num_classes)
        # NOTE: pylint cannot tell that 'sampled_values' is a sequence
        # pylint: disable=unpacking-non-sequence
        sampled, true_expected_count, sampled_expected_count = (
            array_ops.stop_gradient(s) for s in sampled_values)
        # pylint: enable=unpacking-non-sequence
        sampled = math_ops.cast(sampled, dtypes.int64)

        # labels_flat is a [batch_size * num_true] tensor
        # sampled is a [num_sampled] int tensor
        all_ids = array_ops.concat([labels_flat, sampled], 0)

        # weights shape is [num_classes, dim]
        all_w = embedding_ops.embedding_lookup(
            weights, all_ids, partition_strategy=partition_strategy)
        all_b = embedding_ops.embedding_lookup(
            biases, all_ids, partition_strategy=partition_strategy)
        # true_w shape is [batch_size * num_true, dim]
        # true_b is a [batch_size * num_true] tensor
        true_w = array_ops.slice(
            all_w, [0, 0], array_ops.stack([array_ops.shape(labels_flat)[0], -1]))
        true_b = array_ops.slice(all_b, [0], array_ops.shape(labels_flat))

        # inputs shape is [batch_size, dim]
        # true_w shape is [batch_size * num_true, dim]
        # row_wise_dots is [batch_size, num_true, dim]
        dim = array_ops.shape(true_w)[1:2]
        new_true_w_shape = array_ops.concat([[-1, num_true], dim], 0)
        row_wise_dots = math_ops.multiply(
            array_ops.expand_dims(inputs, 1),
            array_ops.reshape(true_w, new_true_w_shape))
        # We want the row-wise dot plus biases which yields a
        # [batch_size, num_true] tensor of true_logits.
        dots_as_matrix = array_ops.reshape(row_wise_dots,
                                           array_ops.concat([[-1], dim], 0))
        true_logits = array_ops.reshape(_sum_rows(dots_as_matrix), [-1, num_true])
        true_b = array_ops.reshape(true_b, [-1, num_true])
        true_logits += true_b

        # Lookup weights and biases for sampled labels.
        #   sampled_w shape is [num_sampled, dim]
        #   sampled_b is a [num_sampled] float tensor
        sampled_w = array_ops.slice(
            all_w, array_ops.stack([array_ops.shape(labels_flat)[0], 0]), [-1, -1])
        sampled_b = array_ops.slice(all_b, array_ops.shape(labels_flat), [-1])

        # inputs has shape [batch_size, dim]
        # sampled_w has shape [num_sampled, dim]
        # sampled_b has shape [num_sampled]
        # Apply X*W'+B, which yields [batch_size, num_sampled]
        sampled_logits = math_ops.matmul(
            inputs, sampled_w, transpose_b=True) + sampled_b

        if remove_accidental_hits:
            acc_hits = candidate_sampling_ops.compute_accidental_hits(
                labels, sampled, num_true=num_true)
            acc_indices, acc_ids, acc_weights = acc_hits

            # This is how SparseToDense expects the indices.
            acc_indices_2d = array_ops.reshape(acc_indices, [-1, 1])
            acc_ids_2d_int32 = array_ops.reshape(
                math_ops.cast(acc_ids, dtypes.int32), [-1, 1])
            sparse_indices = array_ops.concat([acc_indices_2d, acc_ids_2d_int32], 1,
                                              "sparse_indices")
            # Create sampled_logits_shape = [batch_size, num_sampled]
            sampled_logits_shape = array_ops.concat(
                [array_ops.shape(labels)[:1], array_ops.expand_dims(num_sampled, 0)],
                0)
            if sampled_logits.dtype != acc_weights.dtype:
                acc_weights = math_ops.cast(acc_weights, sampled_logits.dtype)
            sampled_logits += sparse_ops.sparse_to_dense(
                sparse_indices,
                sampled_logits_shape,
                acc_weights,
                default_value=0.0,
                validate_indices=False)

        if subtract_log_q:
            # Subtract log of Q(l), prior probability that l appears in sampled.
            true_logits -= math_ops.log(true_expected_count)
            sampled_logits -= math_ops.log(sampled_expected_count)

        # Construct output logits and labels. The true labels/logits start at col 0.
        out_logits = array_ops.concat([true_logits, sampled_logits], 1)
        # true_logits is a float tensor, ones_like(true_logits) is a float tensor
        # of ones. We then divide by num_true to ensure the per-example labels sum
        # to 1.0, i.e. form a proper probability distribution.
        out_labels = array_ops.concat([
            array_ops.ones_like(true_logits) / num_true,
            array_ops.zeros_like(sampled_logits)
        ], 1)

    return out_logits, out_labels


def jean_sampled_softmax_layer(
        nb_classes,
        nb_samples,
        dim,
        labels,  # [B, 1]
        inputs,  # [B, d]
        is_training):
    """
    Creates a sampled softmax layer.

    This layer includes a projection matrix [nb_classes, dim] and bias vector [nb_classes] for classes.
    These class embeddings are used to compare classes to forward input activations.

    :param nb_classes: total number of classes
    :param nb_samples: number of random samples
    :param dim: dimensionality of class embeddings (must match the input dimensionality)
    :param labels: integers indicating the target class [batch_size, 1]
    :param inputs: forward activations [batch_size, dim]
    :param is_training: tf.bool indicating whether training (sampled softmax) or prediction (exact softmax)
    :return:
        logits: [batch_size, S] if training, otherwise [batch_size, V]
        targets: [batch_size]
    """

    # Projection for classes
    # [V, d]
    c_embeddings = tf.get_variable(
        name='class_embedding',
        initializer=tf.random_uniform_initializer(),
        shape=[nb_classes, dim])
    # [V]
    c_biases = tf.get_variable(
        name='class_bias',
        initializer=tf.random_uniform_initializer(),
        shape=[nb_classes])

    def sampled_softmax():
        """
        In training we use a (fast) sampled softmax approximation.

        :return:
            logits: [B, S]
            targets: [B]
        """

        # logits: [B, S]
        # tgtpmf: [B, S]
        logits, tgtpmf = compute_sampled_logits(
            weights=c_embeddings,
            biases=c_biases,
            labels=labels,  # [B, 1]
            inputs=inputs,  # [B, d]
            num_sampled=nb_samples,
            num_classes=nb_classes,
            num_true=1,
            sampled_values=None,  # use this to bypass log_uniform_candidate_sampler
            remove_accidental_hits=True,  # make sure we only sample negative classes
            subtract_log_q=True,
            partition_strategy="div"
        )

        # Targets
        # [B]
        targets = tf.argmax(tgtpmf, axis=-1)

        return logits, targets

    def exact_softmax():
        """
        In validation/test we use the (slow) exact softmax.

        :return:
            logits: [B, V]
            targets: [B]
        """

        # Logits
        # [B, V]
        logits = tf.matmul(
            inputs,  # [B, d]
            tf.transpose(c_embeddings)  # [d, V]
        )
        logits = tf.nn.bias_add(logits, c_biases)

        # Targets
        targets = tf.reshape(labels, [-1])

        return logits, targets

    # depending on training/prediction phase we get approximate/exact logits and targets
    return tf.cond(is_training, true_fn=sampled_softmax, false_fn=exact_softmax)


def compute_sampled_logits_css(
        weights,  # [V, d]
        biases,   # [V]
        probable,   # [B, P]
        inputs,   # [B, T, d]
        num_sampled,
        num_classes,
        minval=1,  # use this to control what's the first valid class
        bias_correction=True
):
    """

    :param weights: class embedding matrix
    :param biases: class biases
    :param probable: probable sequences [B, P]
    :param inputs: forward activations [B, T, d]
    :param num_sampled: number of random uniform samples
    :param num_classes: total number of classes
    :param minval: initial class (defaults to 1)
    :param bias_correction: whether or not to employ Barber's bias correction for negative classes (defaults to True)
    :return:
        logits: [B, T, S]
        support: [B, S]
        The support will include all of the probable classes and it will have no repetitions.
        Supports are sampled for each training sequence (and the entire sequence is considered probable).
    """
    batch_size = tf.shape(inputs)[0]
    # [B, N]
    uniform_samples = tf.random_uniform(
        [batch_size, num_sampled],
        minval=minval, maxval=num_classes, dtype=probable.dtype
    )
    # [B, P + N]
    all_samples = tf.concat([probable, uniform_samples], axis=-1)
    # Sort the samples in the support
    # [B, P + N]
    support, indices = tf.nn.top_k(all_samples, k=tf.shape(all_samples)[-1], sorted=True)
    # Extends the tensor to the left with a column whose elements are larger than those in the original first column
    # [B, 1 + P + N]
    left_augmented = tf.concat([tf.expand_dims(support[:, 0] + 1, -1), support], axis=-1)
    # Create a mask that sets duplicates to zero
    # [B, P + N]
    support_mask = tf.not_equal(left_augmented[:, 1:], left_augmented[:, :-1])
    # Cleans the support as to have non-zero labels appear only once
    # [B, S] where S = P + N
    support *= tf.cast(support_mask, support.dtype)
    # Extends the support with the zero class
    # support = tf.concat([support, tf.zeros([batch_size, 1], dtype=support.dtype)], axis=-1)

    # Embed classes in support
    # [B, S, d]
    c_emb = tf.nn.embedding_lookup(
        weights,  # [V, d]
        support   # [B, S]
    )

    # [B, T, S]
    logits = tf.matmul(
        inputs,  # [B, T, d]
        c_emb,   # [B, S, d]
        transpose_b=True
    )

    # Incorporate class bias
    # [B, S]
    c_bias = tf.nn.embedding_lookup(
        biases,  # [V]
        support  # [B, S]
    )
    # [B, T, S]
    logits += tf.expand_dims(c_bias, 1)  # this makes c_bias [B, 1, S]

    if bias_correction:  # as in (Botev et al, 2017)
        # Incorporate bias correction terms
        # Get the positions in the support where probable labels ended up
        # [B, S]
        probable_indices = tf.less(indices,  # [B, S]
                                   # this is P
                                   tf.shape(probable)[1])
        # Number of unique probable classes
        # [B]
        num_unique_probable = tf.reduce_sum(
            tf.cast(  # cast to float so I can sum
                # mask the probable indices for uniqueness
                tf.logical_and(probable_indices, support_mask),
                tf.float32
            ),
            -1
        )
        # Number of unique negative classes
        # [B]
        num_unique_negative = tf.reduce_sum(
            tf.cast(  # cast to float so I can sum
                # mask the negative indices for uniqueness
                tf.logical_and(
                    # an index that's not in the probable set is in the negative set
                    tf.logical_not(probable_indices),
                    support_mask
                ),
                tf.float32
            ),
            -1
        )
        # Correction term for negative samples
        #  as in section 4 of Botev et al (2017)
        #    \kappa(c) = 1.0 / (|neg| * q(c))
        #  here we use a uniform proposal q(c) over all classes except those in the probable set
        #    q(c) = 1.0 / (V - |pos|)
        #  thus
        #    \kappa(c) = (V - |pos|) / (|neg|)
        # [B]
        log_kappa_neg = tf.log((num_classes - num_unique_probable)) - tf.log(num_unique_negative)
        # We make a kappa matrix for all label (but positive labels get kappa=1)
        # [B, S]
        log_kappa = tf.where(
            probable_indices,  # [B, S]
            tf.zeros(tf.shape(support), dtype=tf.float32),  # no penalties for probable elements
            tf.ones(tf.shape(support), dtype=tf.float32) * tf.expand_dims(log_kappa_neg, 1),  # penalty computed above
        )
        # [B, T, S]
        logits += tf.expand_dims(log_kappa, 1)  # expand log_kappa to have a time dimension

    # Sets logits of duplicate labels to -inf
    # [B, S]
    logits_mask = tf.log(tf.cast(support_mask, logits.dtype) + tf.contrib.keras.backend.epsilon())
    # [B, T, S]
    logits -= tf.expand_dims(logits_mask, 1)  # this makes the mask [B, 1, S]

    # [B, T, S], [B, S]
    return logits, support


def botev_sampled_softmax_layer(
        nb_classes,
        nb_samples,
        dim,
        labels,  # [B, N]
        inputs,  # [B, M, d]
        is_training):
    """
    Creates a sampled softmax layer that can be used for marginalisation.

    :param nb_classes: total number of classes
    :param nb_samples: number of uniform samples
    :param dim: input (and class) dimensionality d
    :param labels: batch of gold sequences [B, N]
    :param inputs: forward activations [B, M, d]
        Here the sequence of activations do not need to have the same length as labels.
    :param is_training: in training we sample the support, in test/validation we use the entire support
    :return:
        logits: [B, M, S]
        targets: [B, N]

        The support has size S and is sampled for each training instance (a sequence).
         For the ith sequence, it includes all classes in labels[i] plus up to `nb_samples` random classes.
         The support does not contain repetitions of non-zero classes.
    """

    # Projection for classes
    # [V, d]
    c_embeddings = tf.get_variable(
        name='class_embedding',
        initializer=tf.random_uniform_initializer(),
        shape=[nb_classes, dim])
    # [V]
    c_biases = tf.get_variable(
        name='class_bias',
        initializer=tf.random_uniform_initializer(),
        shape=[nb_classes])

    def sampled_softmax():
        """
        In training we use a (fast) sampled softmax approximation.

        :return:
            logits: [B, M, S]
            targets: [B, N]
        """

        # logits: [B, M, S]
        # support: [B, S]
        logits, support = compute_sampled_logits_css(
            weights=c_embeddings,
            biases=c_biases,
            probable=labels,  # [B, N]
            inputs=inputs,  # [B, M, d]
            num_sampled=nb_samples,
            num_classes=nb_classes,
        )

        # Targets are the position in the support where the gold labels in the batch occurred
        # we make one-hot vectors where 1s signal the position where the gold label ended up in the support
        # we are guaranteed to find the gold label in the support because the gold label is part of the probable set
        # we are also guaranteed to find it only once, because the support does not have duplicate entries
        # [B, N, S]
        tgtpmf = tf.cast(  # cast to float to make a valid pmf
            tf.equal(  # compare whether a label in the support matches a gold label
                tf.expand_dims(labels, 2),  # make (gold) labels [B, N, 1]
                tf.expand_dims(support, 1)  # make labels in support [B, 1, S]
            ),
            tf.float32
        )

        # Targets
        # [B, N]
        targets = tf.argmax(tgtpmf, axis=-1)

        # [B, M, S], [B, N]
        return logits, targets

    def exact_softmax():
        """
        In validation/test we use the (slow) exact softmax.

        :return:
            logits: [B, M, V]
            targets: [B, N]
        """

        # Logits
        # [B * M, V]
        logits = tf.matmul(
            tf.reshape(inputs, [-1, dim]),  # [B * M, d]
            tf.transpose(c_embeddings)  # [d, V]
        )
        logits = tf.nn.bias_add(logits, c_biases)
        # [B, M, V]
        logits = tf.reshape(logits, [tf.shape(labels)[0], -1, nb_classes])

        # Targets
        # [B, N]
        targets = labels

        return logits, targets

    # depending on training/prediction phase we get approximate/exact logits and targets
    return tf.cond(is_training, true_fn=sampled_softmax, false_fn=exact_softmax)


def np_get_support(
        nb_classes,
        nb_samples,
        labels,
        is_training,
        freq=None
):
    """
    Use numpy to sampled a shared support made of the set of probable classes and a disjoint set of uniformly sampled
    negative classes.

    :param nb_classes: total number of classes C
    :param nb_samples: support size S
    :param labels: [B, T] gold labels
    :param is_training: if True we perform negative sampling, otherwise we use the complete support
    :param freq: [C] use this to sampled from something other than a uniform proposal
    :return: support [S], importance weights [S], number of probable samples P
    """
    if nb_samples >= nb_classes or not is_training:  # no truncation required
        return np.arange(0, nb_classes), np.ones(nb_classes), nb_classes
    # [P]
    probable = np.unique(labels.flatten())
    # P
    nb_probable = probable.shape[0]
    # N
    nb_negative = nb_samples - nb_probable
    if nb_negative <= 0:  # no negative sampling required (probably not a good idea: one should ask enough samples)
        return probable, np.ones(nb_probable), nb_probable
    # Make a proposal
    # [C]
    if freq is None:  # here we have uniform proposal
        q = np.ones(nb_classes)
    else:  # here we base our proposal on freq
        q = np.array(freq)
    # we do not sample probable classes
    q[probable] = 0.
    # renormalise to make it a proper pmf
    q /= q.sum()
    # [N]
    negative = np.random.choice(nb_classes, size=nb_negative, replace=False, p=q)
    # [C]
    importance = np.ones(nb_classes)
    # Adjust importance according to Section 4 of Botev et al 2017
    importance[negative] = 1. / (nb_negative * q[negative])
    # [S]
    support = np.concatenate([probable, negative], -1)
    # [S], [S], P, N
    return support, importance[support], nb_probable


def botev_batch_sampled_softmax_layer(
        nb_classes,
        dim,
        labels,  # [B, N]
        support,  # [S]
        importance,  # [S]
        inputs,  # [B, M, d]
        is_training
):
    """
    Creates a sampled softmax layer that can be used for marginalisation.

    :param nb_classes: total number of classes
    :param dim: input (and class) dimensionality d
    :param labels: batch of gold sequences [B, N]
    :param support: shared support [S]
    :param importance: importance weights of classes in support [S]
    :param inputs: forward activations [B, M, d]
        Here the sequence of activations do not need to have the same length as labels.
    :return:
        logits: [B, M, S]
        targets: [B, N]
    """

    # Projection for classes
    # [V, d]
    c_W = tf.get_variable(
        name='class_embedding',
        initializer=tf.random_uniform_initializer(),
        shape=[nb_classes, dim])
    # [V]
    c_b = tf.get_variable(
        name='class_bias',
        initializer=tf.random_uniform_initializer(),
        shape=[nb_classes])

    def sampled_softmax():
        # Embed classes in support
        # [S, d]
        c_embedded = tf.nn.embedding_lookup(
            c_W,  # [V, d]
            support  # [S]
        )
        # [B, T, S]
        logits = tf.tensordot(
            inputs,  # [B, T, d]
            c_embedded,  # [S, d]
            axes=[[-1], [-1]])

        # Incorporate class bias
        # [S]
        b = tf.nn.embedding_lookup(
            c_b,  # [V]
            support  # [S]
        )
        # [B, T, S]
        logits += tf.expand_dims(tf.expand_dims(b, 0), 0)  # this makes b [1, 1, S]
        # Incorporate the importance weight of the class in the support
        logits += tf.expand_dims(tf.expand_dims(tf.log(importance), 0), 0)  # this makes importance [1, 1, S]

        # Targets are the position in the support where the gold labels in the batch occurred
        # we make one-hot vectors where 1s signal the position where the gold label ended up in the support
        # we are guaranteed to find the gold label in the support because the gold label is part of the probable set
        # we are also guaranteed to find it only once, because the support does not have duplicate entries
        # [B, N, S]
        tgtpmf = tf.cast(  # cast to float to make a valid pmf
            tf.equal(  # compare whether a label in the support matches a gold label
                tf.expand_dims(labels, 2),  # make (gold) labels [B, N, 1]
                tf.expand_dims(tf.expand_dims(support, 0), 0)  # make labels in support [1, 1, S]
            ),
            tf.float32
        )

        # Targets
        # [B, N]
        targets = tf.argmax(tgtpmf, axis=-1)
        return logits, targets

    def exact_softmax():
        """
        In validation/test we use the (slow) exact softmax.

        :return:
            logits: [B, M, V]
            targets: [B, N]
        """

        # Logits
        # [B * M, V]
        logits = tf.matmul(
            tf.reshape(inputs, [-1, dim]),  # [B * M, d]
            tf.transpose(c_W)  # [d, V]
        )
        logits = tf.nn.bias_add(logits, c_b)
        # [B, M, V]
        logits = tf.reshape(logits, [tf.shape(labels)[0], -1, nb_classes])

        # Targets
        # [B, N]
        targets = labels
        return logits, targets

    # depending on training/prediction phase we get approximate/exact logits and targets
    return tf.cond(is_training, true_fn=sampled_softmax, false_fn=exact_softmax)  # logits, targets
