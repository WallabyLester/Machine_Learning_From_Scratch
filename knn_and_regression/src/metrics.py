import numpy as np

def mean_squared_error(estimates, targets):
    """
    Mean squared error measures the average of the square of the errors (the
    average squared difference between the estimated values and what is 
    estimated. The formula is:

    MSE = (1 / n) * sum_{i=1}^{n} (Y_i - Yhat_i)^2

    Args:
        estimates(np.ndarray): the estimated values (should be the same shape as targets)
        targets(np.ndarray): the ground truth values

    Returns:
        MSE(int): mean squared error calculated by above equation 
    """

    MSE_matrix = []
    n = targets.shape[0]
    for i in range(n):
        MSE_matrix.append(np.linalg.norm(estimates[i] - targets[i])**2)
    MSE = sum(MSE_matrix)/n
    return MSE
