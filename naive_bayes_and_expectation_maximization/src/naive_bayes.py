from numpy.core.numeric import NaN
from numpy.lib.type_check import nan_to_num
from src.utils import softmax

import numpy as np
from copy import deepcopy

class NaiveBayes:
    """
    A Naive Bayes classifier for binary data.
    """

    def __init__(self, smoothing=1):
        """
        Args:
            smoothing: controls the smoothing behavior when computing p(x|y).
                If the word "jackpot" appears `k` times across all documents with
                label y=1, we will instead record `k + self.smoothing`. Then
                `p("jackpot" | y=1) = (k + self.smoothing) / Z`, where Z is a
                normalization constant that accounts for adding smoothing to
                all words.
        """
        self.smoothing = smoothing

    def predict(self, X):
        """
        Return the most probable label for each row x of X.
        You should not need to edit this function.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        """
        Using self.p_y and self.p_x_y, compute the probability p(y | x) for each row x of X.
        While you will have used log probabilities internally, the returned array should be
            probabilities, not log probabilities. You may use src.utils.softmax to transform log
            probabilities to probabilities.

        Args:
            X: a data matrix of shape `[n_documents, vocab_size]` on which to predict p(y | x)

        Returns 
            probs: an array of shape `[n_documents, n_labels]` where probs[i, j] contains
                the probability `p(y=j | X[i, :])`. Thus, for a given row of this array,
                sum(probs[i, :]) == 1.
        """
        n_docs, vocab_size = X.shape
        n_labels = 2

        assert hasattr(self, "p_y") and hasattr(self, "p_x_y"), "Model not fit!"
        assert vocab_size == self.vocab_size, "Vocab size mismatch"

        row_probs = X.dot(self.p_x_y)
        p_y = np.array([np.log(1-np.exp(self.p_y[0])), self.p_y[0]])
        probs = (row_probs + p_y)
        probs = softmax(probs, axis=1)

        return probs

    def fit(self, X, y):
        """
        Compute self.p_y and self.p_x_y using the training data.
        You should store log probabilities to avoid underflow.
        This function *should not* use unlabeled data. Wherever y is NaN, that
        label and the corresponding row of X should be ignored.

        self.p_y should contain the marginal probability of each class label.
            Because we are doing binary classification, you may choose
            to represent p_y as a single value representing p(y=1)

        self.p_x_y should contain the conditional probability of each word
            given the class label: p(x | y). This should be an array of shape
            [n_vocab, n_labels].  Remember to use `self.smoothing` to smooth word counts!
            See __init__ for details. If we see M total words across all N documents with
            label y=1, have a vocabulary size of V words, and see the word "jackpot" `k`
            times, then: `p("jackpot" | y=1) = (k + self.smoothing) / (M + self.smoothing *
            V)` Note that `p("jackpot" | y=1) + p("jackpot" | y=0)` will not sum to 1;
            instead, `sum_j p(word_j | y=1)` will sum to 1.

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: None
        """
        n_docs, vocab_size = X.shape
        n_labels = 2
        self.vocab_size = vocab_size

        X = X[np.invert(np.isnan(y))]
        y = y[np.invert(np.isnan(y))]

        X_one = X[y==1]
        y_one = y[y==1]
        X_zero = X[y==0]

        p_y = np.array([np.size(y_one)/len(y)])
        self.p_y = np.log(p_y)

        k_one = np.sum(X_one, axis=0)
        k_zero = np.sum(X_zero, axis=0)
        total_one = np.sum(k_one)
        total_zero = np.sum(k_zero)

        self.p_x_y = np.zeros((self.vocab_size, n_labels))
        self.p_x_y[:,0] = (k_zero + self.smoothing) / (total_zero + self.smoothing * self.vocab_size)
        self.p_x_y[:,1] = (k_one + self.smoothing) / (total_one + self.smoothing * self.vocab_size)
        self.p_x_y = np.log(self.p_x_y)

    def likelihood(self, X, y):
        """
        Using fit self.p_y and self.p_x_y, compute the log likelihood of the data.
            You should use logs to avoid underflow.
            This function should not use unlabeled data. Wherever y is NaN,
            that label and the corresponding row of X should be ignored.

        Recall that the log likelihood of the data can be written:
          `sum_i (log p(y_i) + sum_j log p(x_j | y_i))`

        Note: If the word w appears `k` times in a document, the term
            `p(w | y)` should appear `k` times in the likelihood for that document!

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: the (log) likelihood of the data.
        """
        assert hasattr(self, "p_y") and hasattr(self, "p_x_y"), "Model not fit!"

        n_docs, vocab_size = X.shape
        n_labels = 2

        X_one = X[y==1]
        X_zero = X[y==0]

        p_y_zero = 1 - np.exp(self.p_y[0])
        likelihood1 = np.log(p_y_zero) + np.nansum(X_zero.dot(self.p_x_y[:,0]))
        likelihood2 = self.p_y[0] + np.nansum(X_one.dot(self.p_x_y[:,1]))
        likelihood = likelihood1 + likelihood2

        return likelihood


class NaiveBayesEM(NaiveBayes):
    """
    A NaiveBayes classifier for binary data,
        that uses unlabeled data in the Expectation-Maximization algorithm
    """

    def __init__(self, max_iter=10, smoothing=1):
        """
        Args:
            max_iter: the maximum number of iterations in the EM algorithm,
                where each iteration contains both an E step and M step.
                You should check for convergence after each iterations,
                e.g. with `np.isclose(prev_likelihood, likelihood)`, but
                should terminate after `max_iter` iterations regardless of
                convergence.
            smoothing: controls the smoothing behavior when computing p(x|y).
                If the word "jackpot" appears `k` times across all documents with
                label y=1, we will instead record `k + self.smoothing`. Then
                `p("jackpot" | y=1) = (k + self.smoothing) / Z`, where Z is a
                normalization constant that accounts for adding smoothing to
                all words.
        """
        self.max_iter = max_iter
        self.smoothing = smoothing

    def fit(self, X, y):
        """
        Compute self.p_y and self.p_x_y using the training data.
        You should store log probabilities to avoid underflow.
        This function *should* use unlabeled data within the EM algorithm.

        During the E-step, use the superclass self.predict_proba to
            infer a distribution over the labels for the unlabeled examples.
            Note: you should *NOT* replace the true labels with your predicted
            labels. You can use a `np.where` statement to only update the
            labels where `np.isnan(y)` is True.

        During the M-step, update self.p_y and self.p_x_y, similar to the
            `fit()` call from the NaiveBayes superclass. However, when counting
            words in an unlabeled example to compute p(x | y), instead of the
            binary label y you should use p(y | x).

        For help understanding the EM algorithm, refer to the lectures and
            http://www.cs.columbia.edu/~mcollins/em.pdf
            This PDF is also uploaded to the course website under readings.
            While Figure 1 of this PDF suggests randomly initializing
            p(y) and p(x | y) before your first E-step, please initialize
            all probabilities equally; e.g. if your vocab size is 4, p(x | y=1)
            would be 1/4 for all values of x. This will make it easier to
            debug your code without random variation, and will checked
            in the `test_em_initialization` test case.

        self.p_y should contain the marginal probability of each class label.
            Because we are doing binary classification, you may choose
            to represent p_y as a single value representing p(y=1)

        self.p_x_y should contain the conditional probability of each word
            given the class label: p(x | y). This should be an array of shape
            [n_vocab, n_labels].  Remember to use `self.smoothing` to smooth word counts!
            See __init__ for details. If we have a vocab size of V and we look 
            at i=1...N documents with label probability delta(y = 1 | i) where "jackpot" 
            appears in each document k_i times and each document has M_i total words, 
            then p("jackpot" | y=1) = (sum_i k_i * delta(y=1 | i) + self.smoothing) / 
            (sum_i M_i * delta(y=1 | i ) + V * self.smoothing

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: None
        """
        n_docs, vocab_size = X.shape
        n_labels = 2
        self.vocab_size = vocab_size

        self.p_x_y = np.zeros((self.vocab_size, n_labels))
        self.p_x_y.fill(np.log(1/vocab_size))

        p_y = np.array([1/2])
        self.p_y = np.log(p_y)

        self.delta = np.zeros((n_docs, n_labels))
        iter = 0
        prev_likelihood = 0
        likelihood = 100

        while iter < self.max_iter or (np.isclose(prev_likelihood, likelihood) is False):
            prev_likelihood = deepcopy(likelihood)
            # E-step 
            X_unlabeled = X[np.isnan(y)]
            probs = self.predict_proba(X_unlabeled)

            i = 0
            j = 0
            for value in y:
                if value == 1:
                    self.delta[i][0] = 0
                    self.delta[i][1] = 1
                elif value == 0:
                    self.delta[i][0] = 1
                    self.delta[i][1] = 0
                elif np.isnan(value) == True:
                    self.delta[i][0] = probs[j][0]
                    self.delta[i][1] = probs[j][1]
                    j += 1
                i += 1
            
            # M-step
            p_y = np.array([np.sum(self.delta[:,1])/len(y)])
            self.p_y = np.log(p_y)

            delta_one = np.array([self.delta[:,1]]).T
            delta_zero = np.array([self.delta[:,0]]).T
            
            k_one = np.sum(X.toarray() * delta_one, axis=0)
            k_zero = np.sum(X.toarray() * delta_zero,axis=0)
            M_one = np.sum(k_one)
            M_zero = np.sum(k_zero)
            
            self.p_x_y = np.zeros((self.vocab_size, n_labels))
            self.p_x_y[:,0] = (k_zero + self.smoothing) / (M_zero + self.smoothing * self.vocab_size)
            self.p_x_y[:,1] = (k_one + self.smoothing) / (M_one + self.smoothing * self.vocab_size)
            self.p_x_y = np.log(self.p_x_y)

            likelihood = self.likelihood(X,y)

            iter += 1

    def likelihood(self, X, y):
        """
        Using fit self.p_y and self.p_x_y, compute the likelihood of the data.
            You should use logs to avoid underflow.
            This function *should* use unlabeled data.

        For unlabeled data, we define `delta(y | i) = p(y | x_i)` using the
            previously-learned p(x|y) and p(y) necessary to compute
            that probability. For labeled data, we define `delta(y | i)`
            as 1 if `y_i = y` and 0 otherwise; this is because for labeled data,
            the probability that the ith example has label y_i is 1.
            Following http://www.cs.columbia.edu/~mcollins/em.pdf,
            the log likelihood of the data can be written as:

            `sum_i sum_y (delta(y | i) * (log p(y) + sum_j log p(x_{i,j} | y)))`

        Note: If the word w appears `k` times in a document, the term
            `p(w | y)` should appear `k` times in the likelihood for that document!

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: the (log) likelihood of the data.
        """

        assert hasattr(self, "p_y") and hasattr(self, "p_x_y"), "Model not fit!"

        n_docs, vocab_size = X.shape
        n_labels = 2

        p_y_zero = 1 - np.exp(self.p_y[0])
        likelihood1 = np.sum(self.delta[:,0] * (np.log(p_y_zero) + X.dot(self.p_x_y)[:,0]))
        likelihood2 = np.sum(self.delta[:,1] * (self.p_y[0] + X.dot(self.p_x_y)[:,1]))
        likelihood = likelihood2 + likelihood1

        return likelihood
