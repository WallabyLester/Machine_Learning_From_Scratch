import numpy as np
from numpy.core.fromnumeric import mean
from numpy.core.numeric import True_
from numpy.testing._private.utils import rand
from polynomial_regression import PolynomialRegression
from generate_regression_data import generate_regression_data
from metrics import mean_squared_error  # mse
from math import log  # use if scale too large to see error
from k_nearest_neighbor import KNearestNeighbor
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Number 7, split A

    degree = 4
    N = 100
    x, y = generate_regression_data(degree, N, amount_of_noise=0.1)

    rand_sampl = np.random.choice(N, N, replace=False)  # do not reselect numbers
    x_training, y_training = x[rand_sampl[:10]], y[rand_sampl[:10]]
    x_test, y_test = x[rand_sampl[10:]], y[rand_sampl[10:]]

    plots = []
    mse_training = []
    mse_test = []

    # to 9 degrees 
    for i in range(9):
        poly = PolynomialRegression(i)
        poly.fit(x_training, y_training)
        poly.visualize(x_training, y_training, path=f"../plots_N7_splitA/training_plot_degree_{i}", 
                                               title=f"Training Plot Degree {i}")
        # test will be red
        poly.visualize(x_test, y_test, path=f"../plots_N7_splitA/test_plot_degree_{i}", 
                                               title=f"Test Plot Degree {i}", color='r')
        y_hat_training = poly.predict(x_training)  # predicted value
        mse_training.append(mean_squared_error(y_training, y_hat_training))
        y_hat_test = poly.predict(x_test)
        mse_test.append(mean_squared_error(y_test, y_hat_test))
        plots.append(poly)

    plt.clf()  # clear figure
    plt.figure()
    # log was needed to scale
    plt.plot(range(9), [log(mse_training[i]) for i in range(9)], label="training error")
    plt.plot(range(9), [log(mse_test[i]) for i in range(9)], label="test error")
    plt.title("Error as a Function of Degree")
    plt.xlabel("degree")
    plt.ylabel("error")
    plt.legend()
    plt.grid(True)
    plt.savefig("../plots_N7_splitA/error_as_a_function_of_degree.png")

    # get the two lowest errors
    low_test_err_degree = mse_test.index(min(mse_test))
    low_training_err_degree = mse_training.index(min(mse_training))

    plt.clf()  # clear figure
    plt.figure()
    plt.scatter(x_training, y_training)
    plt.plot(np.sort(plots[low_training_err_degree].X_training), plots[low_training_err_degree].f, label=f"lowest training error curve with degree = {low_training_err_degree}")
    plt.plot(np.sort(plots[low_test_err_degree].X_training), plots[low_test_err_degree].f, label=f"lowest test error curve with degree = {low_test_err_degree}")
    plt.title("Lowest Training and Test Errors")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.savefig("../plots_N7_splitA/lowest_training_and_test_error.png")

    # Number 10, split A

    k = {1, 3, 5, 7, 9}
    kplots = []
    mse_training_k = []
    mse_test_k = []
    kx_training = np.reshape(x_training, (-1,2))
    ky_training = np.reshape(y_training, (-1,2))
    kx_test = np.reshape(x_test, (-1, 2))
    ky_test = np.reshape(y_test, (-1,2))
    #print(kx_training)
    #print(kx_training.shape)

    for i in k:
        knn = KNearestNeighbor(i, distance_measure="euclidean", aggregator="mean")
        knn.fit(kx_training, ky_training)
        #print(f"x_training = {x_training.shape}")
        k_training = knn.predict(kx_training)
        mse_training_k.append(mean_squared_error(ky_training, k_training))
        k_test = knn.predict(kx_test)
        mse_test_k.append(mean_squared_error(ky_test, k_test))
        kplots.append(knn)
    
    plt.clf()  # clear figure
    plt.figure()
    plt.plot(range(5), [(mse_training_k[i]) for i in range(5)], label="training error")
    plt.plot(range(5), [(mse_test_k[i]) for i in range(5)], label="test error")
    plt.title("Error as a Function of k")
    plt.xlabel("k")
    plt.ylabel("error")
    plt.legend()
    plt.grid(True)
    plt.savefig("../plots_N10_splitA/error_as_a_function_of_k.png")

    low_test_err_k = mse_test_k.index(min(mse_test_k))

    plt.clf()  # clear figure
    plt.figure()
    plt.scatter(x_training, y_training)
    plt.plot(np.sort(kplots[low_test_err_k]), kplots[low_test_err_k], label=f"lowest test error curve with k = {low_test_err_k}")
    plt.title("Lowest Test Error")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.savefig("../plots_N10_splitA/lowest_test_error.png")
    
    # Number 9, split B
    
    rand_sampl = np.random.choice(N, N, replace=False)  # do not reselect numbers
    x_training, y_training = x[rand_sampl[:50]], y[rand_sampl[:50]]
    x_test, y_test = x[rand_sampl[50:]], y[rand_sampl[50:]]

    plots = []
    mse_training = []
    mse_test = []

    # to 9 degrees 
    for i in range(9):
        poly = PolynomialRegression(i)
        poly.fit(x_training, y_training)
        poly.visualize(x_training, y_training, path=f"../plots_N9_splitB/training_plot_degree_{i}", 
                                               title=f"Training Plot Degree {i}")
        # test will be red
        poly.visualize(x_test, y_test, path=f"../plots_N9_splitB/test_plot_degree_{i}", 
                                               title=f"Test Plot Degree {i}", color='r')
        y_hat_training = poly.predict(x_training)  # predicted value
        mse_training.append(mean_squared_error(y_training, y_hat_training))
        y_hat_test = poly.predict(x_test)
        mse_test.append(mean_squared_error(y_test, y_hat_test))
        plots.append(poly)

    plt.clf()  # clear figure
    plt.figure()
    # log was needed to scale
    plt.plot(range(9), [log(mse_training[i]) for i in range(9)], label="training error")
    plt.plot(range(9), [log(mse_test[i]) for i in range(9)], label="test error")
    plt.title("Error as a Function of Degree")
    plt.xlabel("degree")
    plt.ylabel("error")
    plt.legend()
    plt.grid(True)
    plt.savefig("../plots_N9_splitB/error_as_a_function_of_degree.png")

    # get the two lowest errors
    low_test_err_degree = mse_test.index(min(mse_test))
    low_training_err_degree = mse_training.index(min(mse_training))

    plt.clf()  # clear figure
    plt.figure()
    plt.scatter(x_training, y_training)
    plt.plot(np.sort(plots[low_training_err_degree].X_training), plots[low_training_err_degree].f, label=f"lowest training error curve with degree = {low_training_err_degree}")
    plt.plot(np.sort(plots[low_test_err_degree].X_training), plots[low_test_err_degree].f, label=f"lowest test error curve with degree = {low_test_err_degree}")
    plt.title("Lowest Training and Test Errors")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.savefig("../plots_N9_splitB/lowest_training_and_test_error.png")

# Number 10, split B

    k = {1, 3, 5, 7, 9}
    kplots = []
    mse_training_k = []
    mse_test_k = []
    kx_training = np.reshape(x_training, (-1,2))
    ky_training = np.reshape(y_training, (-1,2))
    kx_test = np.reshape(x_test, (-1, 2))
    ky_test = np.reshape(y_test, (-1,2))
    #print(kx_training)
    #print(kx_training.shape)

    for i in k:
        knn = KNearestNeighbor(i, distance_measure="euclidean", aggregator="mean")
        knn.fit(kx_training, ky_training)
        #print(f"x_training = {x_training.shape}")
        k_training = knn.predict(kx_training)
        mse_training_k.append(mean_squared_error(ky_training, k_training))
        k_test = knn.predict(kx_test)
        mse_test_k.append(mean_squared_error(ky_test, k_test))
        kplots.append(poly)
    
    plt.clf()  # clear figure
    plt.figure()
    plt.plot(range(5), [(mse_training_k[i]) for i in range(5)], label="training error")
    plt.plot(range(5), [(mse_test_k[i]) for i in range(5)], label="test error")
    plt.title("Error as a Function of k")
    plt.xlabel("k")
    plt.ylabel("error")
    plt.legend()
    plt.grid(True)
    plt.savefig("../plots_N10_splitB/error_as_a_function_of_k.png")

    low_test_err_k = mse_test_k.index(min(mse_test_k))

    plt.clf()  # clear figure
    plt.figure()
    plt.scatter(x_training, y_training)
    plt.plot(np.sort(kplots[low_test_err_k].X_training), kplots[low_test_err_k].f, label=f"lowest test error curve with k = {low_test_err_k}")
    plt.title("Lowest Test Error")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.savefig("../plots_N10_splitB/lowest_test_error.png")
