import numpy as np 

def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.

    Args:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Return:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    a = X.shape[0]
    b = Y.shape[0]
    D = np.zeros(shape=(a,b))
    n = 0
    for i in X:
        m = 0
        for j in Y:
            elem = np.linalg.norm(i-j)
            D[n][m] = elem
            m += 1
        n += 1
    return D

def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.

    Args:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    a = X.shape[0]
    b = Y.shape[0]
    D = np.zeros(shape=(a,b))
    n = 0
    for i in X:
        m = 0
        for j in Y:
            elem = np.linalg.norm(i-j, ord=1)
            D[n][m] = elem
            m += 1
        n += 1
    return D
