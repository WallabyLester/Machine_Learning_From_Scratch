from math import dist
import numpy as np 
from .distances import euclidean_distances, manhattan_distances

def mode(a, axis=0):
    """
    Copied from scipy.stats.mode. 
    https://github.com/scipy/scipy/blob/v1.7.1/scipy/stats/stats.py#L361-L451

    Return an array of the modal (most common) value in the passed array.
    If there is more than one such value, only the smallest is returned.
    The bin-count for the modal bins is also returned.
    Parameters
    ----------
    a : array_like
        n-dimensional array of which to find mode(s).
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over
        the whole array `a`.
    """
    scores = np.unique(np.ravel(a))       # get ALL unique values
    testshape = list(a.shape)
    testshape[axis] = 1
    oldmostfreq = np.zeros(testshape)
    oldcounts = np.zeros(testshape)

    for score in scores:
        template = (a == score)
        counts = np.expand_dims(np.sum(template, axis),axis)
        mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
        oldcounts = np.maximum(counts, oldcounts)
        oldmostfreq = mostfrequent

    return mostfrequent, oldcounts

class KNearestNeighbor():    
    def __init__(self, n_neighbors, distance_measure='euclidean', aggregator='mode'):
        """
        K-Nearest Neighbor is a straightforward algorithm that can be highly
        effective. Training time is...well...is there any training? At test time, labels for
        new points are predicted by comparing them to the nearest neighbors in the
        training data.

        Args:
            n_neighbors {int} -- Number of neighbors to use for prediction.
            distance_measure {str} -- Which distance measure to use. Can be one of
                'euclidean' or 'manhattan'. This is the distance measure
                that will be used to compare features to produce labels. 
            aggregator {str} -- How to aggregate a label across the `n_neighbors` nearest
                neighbors. Can be one of 'mode', 'mean', or 'median'.
        """
        self.n_neighbors = n_neighbors
        self.distance_measure = distance_measure
        self.aggregator = aggregator


    def fit(self, features, targets):
        """Fit features, a numpy array of size (n_samples, n_features). For a KNN, this
        function should store the features and corresponding targets in class 
        variables that can be accessed in the `predict` function. Note that targets can
        be multidimensional! 
        
        Args:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            targets {[type]} -- Target labels for each data point, shape of (n_samples, 
                n_dimensions).
        """
        self.training_features = features
        self.training_targets = targets

    def predict(self, features, ignore_first=False):
        """Predict from features, a numpy array of size (n_samples, n_features) Use the
        training data to predict labels on the test features. For each testing sample, compare it
        to the training samples. Look at the self.n_neighbors closest samples to the 
        test sample by comparing their feature vectors. The label for the test sample
        is determined by aggregating the K nearest neighbors in the training data.

        Note that when using KNN for imputation, the predicted labels are the imputed testing data
        and the shape is (n_samples, n_features).

        Args:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            ignore_first {bool} -- If this is True, then we ignore the closest point
                when doing the aggregation. This is used for collaborative
                filtering, where the closest point is itself and thus is not a neighbor. 
                In this case, we would use 1:(n_neighbors + 1).

        Returns:
            labels {np.ndarray} -- Labels for each data point, of shape (n_samples, n_dimensions). 
            This n_dimensions should be the same as n_dimensions of targets in fit function.
        """
        column = self.training_targets.shape[1]
        row = len(features)
        predictions = np.zeros((row, column))
        for i in range(features.shape[0]):
            nearest_neighbor_index = self.find_nearest_neighbor(features[i], self.training_features, ignore_first)
            #print(nearest_neighbor_index) 
            #print(np.shape(nearest_neighbor_index))
            prediction = self.predictor(nearest_neighbor_index)
            predictions[i] = prediction
            #print(f"predictions = {predictions}")
            #print(f"shape of predictions = {np.shape(predictions)}")
        return predictions
    
    def predictor(self, nearest_neighbor_index):
        #label = []
        values = []
        values = self.training_targets[nearest_neighbor_index][:self.n_neighbors]
        #print(f"values = {values}")
        if self.aggregator == 'mean':
            label = np.mean(values, axis=0)
            label = np.array([label])
        elif self.aggregator == 'median':
            label = np.median(values, axis=0)
            label = np.array([label])
        else:
            label,oldcounts = mode(values, axis=0)
        #print(f"label = {label}")
        #print(f"label dim = {label.ndim}")
        #print(f"shape of label = {np.shape(label)}")
        return label
        
    """```distance_measure``` lets you switch between which distance measure you will
        use to compare data points. The behavior is as follows:

        If 'euclidean', use euclidean_distances, if 'manhattan', use manhattan_distances."""
    def find_nearest_neighbor(self, feature, features, ignore_first):
        #print('feature', feature.shape)
        #print('features', features.shape)   
        feature = feature.reshape((1, features.shape[1]))     
        if self.distance_measure == 'euclidean':
            distances = euclidean_distances(feature, features)
        else:
            distances = manhattan_distances(feature, features)
        #print(f"distance = {distances}", distances.shape)
        return self.find_nearest_index(distances[0], ignore_first)

    def find_nearest_index(self, distances, ignore_first):
         if ignore_first:
            distances_sorted = np.argsort(distances)
            distances_ignore_closest = np.delete(distances_sorted, distances_sorted[0])
            neighbor_index = distances_ignore_closest
         else:
            neighbor_index = np.argsort(distances)
         return neighbor_index

# testing code
'''
if __name__ == "__main__":
    features = np.array([
        [-1, 1, 1, -1, 2],
        [-1, 1, 1, -1, 1],
        [-1, 2, 2, -1, 1],
        [-1, 1, 1, -1, 1],
        [-1, 1, 1, -1, 1]
    ])

    predict = np.array([
        [-1, 1, 0, -1, 0],
        [-1, 1, 1, -1, 0],
        [-1, 0, 1, 0, 0],
        [-1, 1, 1, -1, 1],
        [-1, 1, 1, -1, 0]
    ])
    targets = np.array([
        [1, 0, 1],
        [1, 1, 5],
        [3, 1, 1],
        [1, 1, 2],
        [5, 1, 1]
    ])

    knn = KNearestNeighbor(1, distance_measure="euclidean", aggregator="mean")
    knn.fit(features, targets)

    # predict and calculate accuracy
    labels = knn.predict(predict)
'''