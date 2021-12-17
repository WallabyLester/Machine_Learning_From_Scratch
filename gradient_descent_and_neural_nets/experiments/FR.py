from scipy.optimize.nonlin import TerminationCondition
from your_code import L1Regularization, L2Regularization, HingeLoss, SquaredLoss, ZeroOneLoss
from your_code import GradientDescent, accuracy, confusion_matrix
from your_code import load_data
import numpy as np 
import matplotlib.pyplot as plt

# Problem 1a
train_features, _, train_targets, _ = load_data('mnist-binary', fraction=1.0)

gd = GradientDescent(loss='hinge', learning_rate=1e-4)
gd.fit(train_features, train_targets)

plt.figure()
plt.plot(range(len(gd.loss_list)), gd.loss_list)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss by Iterations')
plt.savefig('experiments/Q1a_loss.png')

plt.figure()
plt.plot(range(len(gd.acc_list)), gd.acc_list)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Accuracy by Iterations')
plt.savefig('experiments/Q1a_accuracy.png')

# Problem 1b
train_features, _, train_targets, _ = load_data('mnist-binary', fraction=1.0)

gd = GradientDescent(loss='hinge', learning_rate=1e-4)
gd.fit(train_features, train_targets, batch_size=64, max_iter=1000*train_features.shape[0])

plt.figure()
plt.plot(range(len(gd.loss_list)), gd.loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss by Epoch')
plt.savefig('experiments/Q1b_loss.png')

plt.figure()
plt.plot(range(len(gd.acc_list)), gd.acc_list)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy by Epoch')
plt.savefig('experiments/Q1b_accuracy.png')

# Problem 2a
train_features, _, train_targets, _ = load_data('synthetic', fraction=1.0)

bias_list = np.linspace(-5.5, 0.5, 100)
loss_list = []

w0 = np.ones(train_features.shape[1])
bias = np.ones((len(train_features), 1))
train_features = np.hstack((train_features, bias))

for bias in bias_list:
    z0l = ZeroOneLoss()
    w = np.append(w0, bias)
    loss = z0l.forward(train_features, w, train_targets)
    loss_list.append(loss)

plt.figure()
plt.plot(bias_list, loss_list)
plt.xlabel('Bias')
plt.ylabel('Loss')
plt.title('Loss Landscape')
plt.savefig('experiments/Q2a_loss_landscape.png')

# Problem 2b
train_features, _, train_targets, _ = load_data('synthetic', fraction=1.0)

train_features = train_features[[0, 1, 3, 4]]
train_targets = train_targets[[0, 1, 3, 4]]

bias_list = np.linspace(-5.5, 0.5, 100)
loss_list = []

w0 = np.ones(train_features.shape[1])
bias = np.ones((len(train_features), 1))
train_features = np.hstack((train_features, bias))

for bias in bias_list:
    z0l = ZeroOneLoss()
    w = np.append(w0, bias)
    loss = z0l.forward(train_features, w, train_targets)
    loss_list.append(loss)

plt.figure()
plt.plot(bias_list, loss_list)
plt.xlabel('Bias')
plt.ylabel('Loss')
plt.title('Loss Landscape')
plt.savefig('experiments/Q2b_loss_landscape.png')

# Problem 3a
train_features, test_features, train_targets, test_targets = load_data('mnist-binary', fraction=1.0)

lam_list = [1e-3, 1e-2, 1e-1, 1, 10, 100]
l1_list = []
l2_list = []
epsilon = 1e-3

for lam in lam_list:
    l1 = GradientDescent('squared', regularization='l1', learning_rate=1e-5, reg_param=lam)
    l2 = GradientDescent('squared', regularization='l2', learning_rate=1e-5, reg_param=lam)

    l1.fit(train_features, train_targets, max_iter=2000)
    l2.fit(train_features, train_targets, max_iter=2000)

    l1_list.append(np.sum(np.where(abs(l1.model) > epsilon, 1, 0)))
    l2_list.append(np.sum(np.where(abs(l2.model) > epsilon, 1, 0)))

plt.figure()
plt.plot(range(len(lam_list)), l1_list, label='l1')
plt.plot(range(len(lam_list)), l2_list, label='l2')
plt.xlabel('Lambda')
plt.ylabel('Weights > Epsilon')
plt.title('Regularizers')
plt.legend()
plt.xticks([0,1,2,3,4,5], ['1e-3', '1e-2', '1e-1', '1', '10', '100'])
plt.savefig('experiments/Q4a_regularizers.png')
