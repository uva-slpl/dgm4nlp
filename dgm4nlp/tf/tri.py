"""Utilities for probability distributions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import hashlib
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn

def fill_triangular(x, upper=False, name=None):
  """Creates a (batch of) triangular matrix from a vector of inputs.
  Created matrix can be lower- or upper-triangular. (It is more efficient to
  create the matrix as upper or lower, rather than transpose.)
  Triangular matrix elements are filled in a clockwise spiral. See example,
  below.
  If `x.get_shape()` is `[b1, b2, ..., bK, d]` then the output shape is `[b1,
  b2, ..., bK, n, n]` where `n` is such that `d = n(n+1)/2`, i.e.,
  `n = int(np.sqrt(0.25 + 2. * m) - 0.5)`.
  Example:
  ```python
  fill_triangular([1, 2, 3, 4, 5, 6])
  # ==> [[4, 0, 0],
  #      [6, 5, 0],
  #      [3, 2, 1]]
  fill_triangular([1, 2, 3, 4, 5, 6], upper=True)
  # ==> [[1, 2, 3],
  #      [0, 5, 6],
  #      [0, 0, 4]]
  ```
  For comparison, a pure numpy version of this function can be found in
  `util_test.py`, function `_fill_triangular`.
  Args:
    x: `Tensor` representing lower (or upper) triangular elements.
    upper: Python `bool` representing whether output matrix should be upper
      triangular (`True`) or lower triangular (`False`, default).
    name: Python `str`. The name to give this op.
  Returns:
    tril: `Tensor` with lower (or upper) triangular elements filled from `x`.
  Raises:
    ValueError: if `x` cannot be mapped to a triangular matrix.
  """

  with ops.name_scope(name, "fill_triangular", values=[x]):
    x = ops.convert_to_tensor(x, name="x")
    if x.shape.with_rank_at_least(1)[-1].value is not None:
      # Formula derived by solving for n: m = n(n+1)/2.
      m = np.int32(x.shape[-1].value)
      n = np.sqrt(0.25 + 2. * m) - 0.5
      if n != np.floor(n):
        raise ValueError("Input right-most shape ({}) does not "
                         "correspond to a triangular matrix.".format(m))
      n = np.int32(n)
      static_final_shape = x.shape[:-1].concatenate([n, n])
    else:
      m = array_ops.shape(x)[-1]
      # For derivation, see above. Casting automatically lops off the 0.5, so we
      # omit it.  We don't validate n is an integer because this has
      # graph-execution cost; an error will be thrown from the reshape, below.
      n = math_ops.cast(
          math_ops.sqrt(0.25 + math_ops.cast(2 * m, dtype=dtypes.float32)),
          dtype=dtypes.int32)
      static_final_shape = x.shape.with_rank_at_least(1)[:-1].concatenate(
          [None, None])
    # We now concatenate the "tail" of `x` to `x` (and reverse one of them).
    #
    # We do this based on the insight that the input `x` provides `ceil(n/2)`
    # rows of an `n x n` matrix, some of which will get zeroed out being on the
    # wrong side of the diagonal. The first row will not get zeroed out at all,
    # and we need `floor(n/2)` more rows, so the first is what we omit from
    # `x_tail`. If we then stack those `ceil(n/2)` rows with the `floor(n/2)`
    # rows provided by a reversed tail, it is exactly the other set of elements
    # of the reversed tail which will be zeroed out for being on the wrong side
    # of the diagonal further up/down the matrix. And, in doing-so, we've filled
    # the triangular matrix in a clock-wise spiral pattern. Neat!
    #
    # Try it out in numpy:
    #  n = 3
    #  x = np.arange(n * (n + 1) / 2)
    #  m = x.shape[0]
    #  n = np.int32(np.sqrt(.25 + 2 * m) - .5)
    #  x_tail = x[(m - (n**2 - m)):]
    #  np.concatenate([x_tail, x[::-1]], 0).reshape(n, n)  # lower
    #  # ==> array([[3, 4, 5],
    #               [5, 4, 3],
    #               [2, 1, 0]])
    #  np.concatenate([x, x_tail[::-1]], 0).reshape(n, n)  # upper
    #  # ==> array([[0, 1, 2],
    #               [3, 4, 5],
    #               [5, 4, 3]])
    #
    # Note that we can't simply do `x[..., -(n**2 - m):]` because this doesn't
    # correctly handle `m == n == 1`. Hence, we do nonnegative indexing.
    # Furthermore observe that:
    #   m - (n**2 - m)
    #   = n**2 / 2 + n / 2 - (n**2 - n**2 / 2 + n / 2)
    #   = 2 (n**2 / 2 + n / 2) - n**2
    #   = n**2 + n - n**2
    #   = n
    if upper:
      x_list = [x, array_ops.reverse(x[..., n:], axis=[-1])]
    else:
      x_list = [x[..., n:], array_ops.reverse(x, axis=[-1])]
    new_shape = (
        static_final_shape.as_list()
        if static_final_shape.is_fully_defined()
        else array_ops.concat([array_ops.shape(x)[:-1], [n, n]], axis=0))
    x = array_ops.reshape(array_ops.concat(x_list, axis=-1), new_shape)
    x = array_ops.matrix_band_part(
        x,
        num_lower=(0 if upper else -1),
        num_upper=(-1 if upper else 0))
    x.set_shape(static_final_shape)
    return x
def tridiag(below=None, diag=None, above=None, name=None):
  """Creates a matrix with values set above, below, and on the diagonal.
  Example:
  ```python
  tridiag(below=[1., 2., 3.],
          diag=[4., 5., 6., 7.],
          above=[8., 9., 10.])
  # ==> array([[  4.,   8.,   0.,   0.],
  #            [  1.,   5.,   9.,   0.],
  #            [  0.,   2.,   6.,  10.],
  #            [  0.,   0.,   3.,   7.]], dtype=float32)
  ```
  Warning: This Op is intended for convenience, not efficiency.
  Args:
    below: `Tensor` of shape `[B1, ..., Bb, d-1]` corresponding to the below
      diagonal part. `None` is logically equivalent to `below = 0`.
    diag: `Tensor` of shape `[B1, ..., Bb, d]` corresponding to the diagonal
      part.  `None` is logically equivalent to `diag = 0`.
    above: `Tensor` of shape `[B1, ..., Bb, d-1]` corresponding to the above
      diagonal part.  `None` is logically equivalent to `above = 0`.
    name: Python `str`. The name to give this op.
  Returns:
    tridiag: `Tensor` with values set above, below and on the diagonal.
  Raises:
    ValueError: if all inputs are `None`.
  """

  def _pad(x):
    """Prepends and appends a zero to every vector in a batch of vectors."""
    shape = array_ops.concat([array_ops.shape(x)[:-1], [1]], axis=0)
    z = array_ops.zeros(shape, dtype=x.dtype)
    return array_ops.concat([z, x, z], axis=-1)

  def _add(*x):
    """Adds list of Tensors, ignoring `None`."""
    s = None
    for y in x:
      if y is None:
        continue
      elif s is None:
        s = y
      else:
        s += y
    if s is None:
      raise ValueError("Must specify at least one of `below`, `diag`, `above`.")
    return s

  with ops.name_scope(name, "tridiag", [below, diag, above]):
    if below is not None:
      below = ops.convert_to_tensor(below, name="below")
      below = array_ops.matrix_diag(_pad(below))[..., :-1, 1:]
    if diag is not None:
      diag = ops.convert_to_tensor(diag, name="diag")
      diag = array_ops.matrix_diag(diag)
    if above is not None:
      above = ops.convert_to_tensor(above, name="above")
      above = array_ops.matrix_diag(_pad(above))[..., 1:, :-1]
    # TODO(jvdillon): Consider using scatter_nd instead of creating three full
    # matrices.
    return _add(below, diag, above)
