from numpy.core.fromnumeric import std
from sklearn.base import TransformerMixin
from your_code.load_data import load_mnist_data, _load_mnist
import warnings
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from numpy import mean, std


# load data with fraction 0.8
train_features, test_features, train_targets, test_targets = load_mnist_data(fraction=0.8)
# print(train_features)
# print(train_targets)
# print(test_features)
# print(test_targets)

# mlp = MLPClassifier(hidden_layer_sizes=(1,))

# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
#     mlp.fit(train_features, train_targets)

# print("Training set score: %f" % mlp.score(train_features, train_targets))

def neural(layer, function, value):
    """ Function to perform MLP

    Takes in hidden layer size and perform MPL on the loaded data set 

    Args:
        layer (int) : hidden layer size

    Returns:
        acc (float) : test accuracy
    
    """
    mlp = MLPClassifier(hidden_layer_sizes=(layer,), activation=function, alpha=value)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
        mlp.fit(train_features, train_targets)
    acc = mlp.score(test_features, test_targets)
    
    return acc

# # Network 1 - 1 hidden layer
# acc_list = []
# for i in range(10):
#     acc = neural(1)
#     acc_list.append(acc)

# net1_mean = mean(acc_list)
# net1_std = std(acc_list)
# print(net1_mean)
# print(net1_std)

# # Network 2 - 4 hidden layers
# acc_list = []
# for i in range(10):
#     acc = neural(4)
#     acc_list.append(acc)

# net2_mean = mean(acc_list)
# net2_std = std(acc_list)
# print(net2_mean)
# print(net2_std)

# # Network 3 - 16 hidden layers
# acc_list = []
# for i in range(10):
#     acc = neural(16)
#     acc_list.append(acc)

# net3_mean = mean(acc_list)
# net3_std = std(acc_list)
# print(net3_mean)
# print(net3_std)

# # Network 4 - 64 hidden layers
# acc_list = []
# for i in range(10):
#     acc = neural(64)
#     acc_list.append(acc)

# net4_mean = mean(acc_list)
# net4_std = std(acc_list)
# print(net4_mean)
# print(net4_std)

# # Network 5 - 256 hidden layers 
# acc_list = []
# for i in range(10):
#     acc = neural(256)
#     acc_list.append(acc)

# net5_mean = mean(acc_list)
# net5_std = std(acc_list)
# print(net5_mean)
# print(net5_std)

# Network 5 - 256 hidden layers activation test 
# identity
# acc_list = []
# for i in range(10):
#     acc = neural(256, 'identity')
#     acc_list.append(acc)

# net5_mean = mean(acc_list)
# net5_std = std(acc_list)
# print(net5_mean)
# print(net5_std)

# # logistic
# acc_list = []
# for i in range(10):
#     acc = neural(256, 'logistic')
#     acc_list.append(acc)

# net5_mean = mean(acc_list)
# net5_std = std(acc_list)
# print(net5_mean)
# print(net5_std)

# # tanh
# acc_list = []
# for i in range(10):
#     acc = neural(256, 'tanh')
#     acc_list.append(acc)

# net5_mean = mean(acc_list)
# net5_std = std(acc_list)
# print(net5_mean)
# print(net5_std)

# # relu
# acc_list = []
# for i in range(10):
#     acc = neural(256, 'relu')
#     acc_list.append(acc)

# net5_mean = mean(acc_list)
# net5_std = std(acc_list)
# print(net5_mean)
# print(net5_std)

# Network 5, relu - test of alpha
# 1
# acc_list = []
# for i in range(10):
#     acc = neural(256, 'relu', 1.0)
#     acc_list.append(acc)

# net5_mean = mean(acc_list)
# net5_std = std(acc_list)
# print(net5_mean)
# print(net5_std)

# # 0.1
# acc_list = []
# for i in range(10):
#     acc = neural(256, 'relu', 0.1)
#     acc_list.append(acc)

# net5_mean = mean(acc_list)
# net5_std = std(acc_list)
# print(net5_mean)
# print(net5_std)

# # 0.01
# acc_list = []
# for i in range(10):
#     acc = neural(256, 'relu', 0.01)
#     acc_list.append(acc)

# net5_mean = mean(acc_list)
# net5_std = std(acc_list)
# print(net5_mean)
# print(net5_std)

# # 0.001
# acc_list = []
# for i in range(10):
#     acc = neural(256, 'relu', 0.001)
#     acc_list.append(acc)

# net5_mean = mean(acc_list)
# net5_std = std(acc_list)
# print(net5_mean)
# print(net5_std)

# # 0.0001
# acc_list = []
# for i in range(10):
#     acc = neural(256, 'relu', 0.0001)
#     acc_list.append(acc)

# net5_mean = mean(acc_list)
# net5_std = std(acc_list)
# print(net5_mean)
# print(net5_std)


# Visualization of optimal network
if __name__ == "__main__":
    """ Calling main function to visualize single time
    """
    mlp = MLPClassifier(hidden_layer_sizes=(256,), activation='relu', alpha=0.01)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
        mlp.fit(train_features, train_targets)

fig, axes = plt.subplots(6,6)
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28,28), cmap=plt.cm.gray, vmin=0.5 * vmin, vmax=0.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()