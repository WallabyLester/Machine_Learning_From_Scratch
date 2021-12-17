import numpy as np 

def generate_regression_data(degree, N, amount_of_noise=1.0):
    """
    Generates data to test one-dimensional regression models.

    Args:
        degree (int): degree of polynomial that relates the output x and y
        N (int): number of points to generate
        amount_of_noise (float): amount of random noise to add to the relationship between x and y.

    Returns:
        x (np.ndarray): explanatory variable of size N, ranges between -1 and 1.
        y (np.ndarray): response variable of size N, which responds to x as polynomial of degree 'degree'.
    """
        
    # Generate explanatory variable x of floats chosen are random between -1 and 1
    x = np.zeros((N,), dtype=float)
    iteration = 0
    for i in x:
        x[iteration] = np.random.uniform(-1.0, 1.0)
        iteration += 1
  
    # Generate polynomial float coefficients chosen uniformally at random between -10 and 10
    exps = np.arange(degree+1)
    coeff = np.zeros((degree+1,), dtype=float)
    j = 0
    for i in coeff:
        coeff[j] = np.random.uniform(-10.0, 10.0)
        j += 1

    # Generate response variable y that contains f(x)
    y = np.zeros(len(x))
    f = 0
    for i in x:
        y[f] = sum(coeff[a]*x[f]**exps[a] for a in range(len(coeff)))
        f += 1

    # Add Gaussian noise n to y with mean = 0.0 and standard deviation
    std_y = np.std(y)
    noise = np.random.normal(loc=0.0, scale=amount_of_noise*std_y, size=y.shape)
    y_noise = y + noise
 
    return x, y_noise






