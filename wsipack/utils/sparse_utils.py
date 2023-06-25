import numpy as np
import scipy

from wsipack.utils.cool_utils import is_list_or_tuple

def _array_row_metric(x, metric_fct, unravel1=True, **kwargs):
    if is_list_or_tuple(x):
        x = np.array(x)
    elif unravel1 and (len(x.shape) == 1 or x.shape[0] == 1 or x.shape[1] == 1):
        x = x.ravel()

    if len(x.shape) == 2:
        result = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            result[i] = metric_fct(x[i], **kwargs)
        return result
    else:
        return metric_fct(x, **kwargs)

def _entropy(x, normalized=True):
    entro = scipy.stats.entropy(x)
    if normalized:
        entro = entro / np.log(len(x))
    return entro

def entropy(x, unravel1=True, normalized=True):
    """ entropy with natural log. """
    return _array_row_metric(x, metric_fct=_entropy, unravel1=unravel1, normalized=normalized)

def _hoyer(x):
    n = float(len(x))
    n_root = n ** (0.5)
    nominator = n_root - sum(abs(x)) / np.linalg.norm(x)
    denominator = n_root - 1
    hoyer = nominator / denominator
    return hoyer


def hoyer(x, unravel1=True):
    """ Sparseness measure from 'Non-negative Matrix Factorization with Sparseness Constraints, 2004'
     0: uniform, 1: sparse """
    return _array_row_metric(x, metric_fct=_hoyer, unravel1=unravel1)
