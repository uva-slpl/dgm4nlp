"""
:Authors: - Wilker Aziz
"""
import numpy as np


def create_synthetic_data(path, nb_samples=10000, vocab_size=10, length_range=[2, 20], cat_params=None, alpha=None):
    """

    :param path: where to save the data
    :param nb_samples: number of sequences
    :param vocab_size: number of known tokens [1, vocab_size]
    :param length_range: shortest and longest sequence
    :param cat_params: distribution over tokens (defaults to uniform if alpha is None, otherwise,
        it will be sampled from a Dirichlet(alpha))
    :param alpha: Dirichlet parameter (defaults to None)
    """

    if cat_params is None:
        if alpha is None:
            cat_params = np.full(vocab_size, 1.0 / vocab_size)  # uniform distribution
        else:
            cat_params = np.random.dirichlet(np.full(vocab_size, alpha))  # sampled from a symmetric Dirichlet
    # sample random sequences
    data = np.random.choice(vocab_size, size=(nb_samples, length_range[1]), p=cat_params) + 1  # 1-based vocab entries
    lengths = np.random.randint(length_range[0], length_range[1] + 1, size=nb_samples)
    # save to disk as text
    with open(path, 'w') as fo:
        for row, length in zip(data, lengths):
            print(' '.join(str(x) for x in row[:length]), file=fo)




