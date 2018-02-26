"""
:Authors: - Wilker Aziz
"""
import gzip
from io import TextIOWrapper
from itertools import zip_longest


def smart_ropen(path):
    """Opens files directly or through gzip depending on extension."""
    if path.endswith('.gz'):
        return TextIOWrapper(gzip.open(path, 'rb'))
    else:
        return open(path, 'r', encoding='utf8')


def smart_wopen(path):
    """Opens files directly or through gzip depending on extension."""
    if path.endswith('.gz'):
        return TextIOWrapper(gzip.open(path, 'wb'))
    else:
        return open(path, 'w', encoding='utf8')


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)