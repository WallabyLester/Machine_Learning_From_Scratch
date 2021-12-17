import numpy as np
from numpy.core.fromnumeric import mean
from numpy.core.numeric import True_
from numpy.testing._private.utils import rand
from polynomial_regression import PolynomialRegression
from generate_regression_data import generate_regression_data
from metrics import mean_squared_error  # mse
from math import log  # use if scale too large to see error
from k_nearest_neighbor import KNearestNeighbor
import os
from load_json_data import load_json_data
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

if __name__ == "__main__":
    features, targets = load_json_data("../data/clean-spiral.json")
    #targets = targets[:, None]
    kx_test = np.reshape(features, (25,4))
    ky_test = np.reshape(targets, (-1,2))

    k = [5]
    kplots = []
    mse_test_k = []

    for i in k:
        knn = KNearestNeighbor(i, distance_measure="euclidean", aggregator="mean")
        knn.fit(kx_test, ky_test)
        k_training = knn.predict(kx_test)
        mse_test_k.append(mean_squared_error(ky_test, k_training))
        kplots.append(knn)

    low_test_err_k = mse_test_k.index(min(mse_test_k))

    plt.clf()  # clear figure
    plt.figure(figsize=(6, 4))
    plt.scatter(features[:, 0], features[:, 1], c=targets)

    plt.plot(np.sort(kx_test)[:,0], k_training[:,0], label=f"lowest test error curve with k = {low_test_err_k}")

    plt.title("clean_spiral")
    plt.savefig("../plots_N11/clean_spiral.png")
    
    features, targets = load_json_data("../data/noisy-linear.json")
    #targets = targets[:, None]
    kx_test = np.reshape(features, (31,4))
    ky_test = np.reshape(targets, (-1,2))

    k = [5]
    kplots = []
    mse_test_k = []

    for i in k:
        knn = KNearestNeighbor(i, distance_measure="euclidean", aggregator="mean")
        knn.fit(kx_test, ky_test)
        k_training = knn.predict(kx_test)
        mse_test_k.append(mean_squared_error(ky_test, k_training))
        kplots.append(knn)

    low_test_err_k = mse_test_k.index(min(mse_test_k))

    plt.clf()  # clear figure
    plt.figure(figsize=(6, 4))
    plt.scatter(features[:, 0], features[:, 1], c=targets)

    plt.plot(np.sort(kx_test)[:,0], k_training[:,0], label=f"lowest test error curve with k = {low_test_err_k}")

    plt.title("noisy_linear")
    plt.savefig("../plots_N11/noisy_linear.png")
    