import numpy as np


def softmax(x, axis):
    """
    Implements a *stabilized* softmax along the correct index
    https://www.deeplearningbook.org/contents/numerical.html

    Do not use scipy to implement this function!
    """
    x = np.atleast_2d(x)

    z = x - np.vstack(np.max(x, axis))
    softmax = np.exp(z) 
    softmax[:,0] = softmax[:,0] / np.sum(np.exp(z),axis)
    softmax[:,1] = softmax[:,1] / np.sum(np.exp(z),axis)

    return softmax
