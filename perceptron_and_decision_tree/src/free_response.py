from metrics import compute_precision_and_recall, compute_confusion_matrix
from metrics import compute_f1_measure, compute_accuracy
from data import load_data, train_test_split
import numpy as np 
import copy
from visualize import plot_decision_regions
try:
    import matplotlib
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


#####################################################################################################
def transform_data(features):
    
    # use euclidean distance 
    origin = np.array((0,0))
    transformed_features = np.linalg.norm(features - origin, axis=1)
    transformed_features = np.vstack(transformed_features)

    return transformed_features

class Perceptron():
    def __init__(self, max_iterations=200):
        
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        
    def classification_function(self, x):
        return np.where(x>0, 1, -1)

    def fit(self, features, targets):
        
        self.ones = np.ones([features.shape[0],1])
        features  = np.hstack((self.ones, features))

        # Relabel targets to (-1, 1) from (0, 1)
        targets = np.where(targets == 0, -1, targets)

        n_samples, n_features = features.shape

        self.weights = np.ones(n_features)
        compare = None
        norm = 1
        i = 0
        while i < self.max_iterations and norm >= 0.01:
            i += 1
            compare = copy.deepcopy(self.weights)
            for index, value in enumerate(features):
                linear_output = np.dot(self.weights, value)*targets[index]
                prediction = self.classification_function(linear_output)

                if prediction < 0:
                    self.weights += value*targets[index]

                else:
                    continue
               
            norm = np.linalg.norm(compare - self.weights)

        print(f"iterations: {i}") 
            

    def predict(self, features):
        
        self.ones = np.ones([features.shape[0],1])
        features  = np.hstack((self.ones, features))
        linear_output = np.dot(features, self.weights) 
        predictions = self.classification_function(linear_output)

        # Relabel predictions back to (0, 1) from (-1, 1)
        predictions = np.where(predictions == -1, 0, predictions)
        return predictions
###################################################################################################################

fraction = 1.0
data_path = 'data/transform_me.csv'
features, targets, attribute_names = load_data(data_path)
# features = transform_data(features)
train_features, train_targets, test_features, test_targets = train_test_split(features, targets, fraction)

perceptron = Perceptron()
perceptron.fit(train_features, train_targets)
predictions = perceptron.predict(test_features)
confusion_matrix = compute_confusion_matrix(test_targets, predictions)
accuracy = compute_accuracy(test_targets, predictions)
precision, recall = compute_precision_and_recall(test_targets, predictions)
f1_measure = compute_f1_measure(test_targets, predictions)

print(f"accuracy of perceptron: {accuracy}")
# print(f"precision: {precision}")
# print(f"recall: {recall}")
# print(f"f1 measure: {f1_measure}")


# fraction = 1.0
# data_path = 'data/transform_me.csv'
# features, targets, attribute_names = load_data(data_path)
# train_features, train_targets, test_features, test_targets = train_test_split(features, targets, fraction)

perceptron = Perceptron()
perceptron.fit(train_features, train_targets)
plot_decision_regions(test_features, test_targets, perceptron, 'Perceptron Decision Regions for Transform Me')
plt.show()
# plt.savefig("perceptron_plots/Perceptron_Decision_Regions_for_Transform_Me.png")

####################################################################################################################
class Node():
    def __init__(self, value=None, attribute_name="root", attribute_index=None, branches=None):
        
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value

class DecisionTree():
    def __init__(self, attribute_names):
        
        self.attribute_names = attribute_names
        self.tree = None

    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )

    def fit_recursion(self, features, targets, curr_node, attributes):
        """ Function to fit data to branch or leaf and discern left or right children 
        
        """
        if features.size == 0 and targets.size == 0:
            return Node(attribute_name='leaf')

        elif np.count_nonzero(targets == 1) == len(targets):
            return Node(attribute_name='leaf', value=1)

        elif np.count_nonzero(targets == 0) == len(targets):
            return Node(attribute_name='leaf', value=0)

        elif len(attributes) == 0:
            if np.count_nonzero(targets == 1) > np.count_nonzero(targets == 0):
                return Node(attribute_name='leaf', value=1)
            else: 
                return Node(attribute_name='leaf', value=0)

        else: 
            values = np.hstack((features, np.vstack(targets))) 
            gain = {}
            for i in range(len(attributes)):
                gain[attributes[i]] = information_gain(features, i ,targets)

            curr_node.attribute_name = max(gain, key=gain.get)
            curr_node.attribute_index = self.attribute_names.index(curr_node.attribute_name)
            curr_idx = attributes.index(curr_node.attribute_name)
            removed = attributes.copy()
            removed.remove(curr_node.attribute_name)
            
            #  S(A < m) and S(A >= m)
            
            if np.any(values[values[:, curr_idx] > 1]) or np.any(values[values[:, curr_idx] < 0]):
                median = np.median(values[:, curr_idx])
                curr_node.value = median
                left_values = values[values[:,curr_idx] < median]
                left_feats = left_values[:,:-1]
                left_feats = np.delete(left_feats, curr_idx, 1)
                left_tgts = left_values[:, -1]
            else: 
                curr_node.value = 1
                left_values = values[values[:, curr_idx] == 0]
                left_feats = left_values[:,:-1]
                left_feats = np.delete(left_feats, curr_idx, 1)
                left_tgts = left_values[:, -1]
            
    
            left = self.fit_recursion(left_feats, left_tgts, Node(), removed)
            
            if np.any(values[values[:, curr_idx] > 1]) or np.any(values[values[:, curr_idx] < 0]):
                median = np.median(values[:, curr_idx])
                curr_node.value = median
                right_values = values[values[:,curr_idx] >= median]
                right_feats = right_values[:,:-1]
                right_feats = np.delete(right_feats, curr_idx, 1)
                right_tgts = right_values[:, -1]
            else:
                curr_node.value = 1
                right_values = values[values[:, curr_idx] == 1]
                right_feats = right_values[:,:-1]
                right_feats = np.delete(right_feats, curr_idx, 1)
                right_tgts = right_values[:, -1]
            

            right = self.fit_recursion(right_feats, right_tgts, Node(), removed)

            curr_node.branches.append(left)
            curr_node.branches.append(right)

            return curr_node            

    def fit(self, features, targets):
        
        self._check_input(features)
        self.tree = self.fit_recursion(features, targets, Node(), self.attribute_names)

    def predictor(self, point, curr_node):
        if curr_node.branches == []:
            return curr_node.value

        else:
            if (point[curr_node.attribute_index] < curr_node.value):
                return self.predictor(point, curr_node.branches[0])
            else:
                return self.predictor(point, curr_node.branches[1])

    def predict(self, features):
        
        self._check_input(features)
        predictions = np.zeros((features.shape[0]))

        iter = 0
        for j in features:
            prediction = self.predictor(j, self.tree)
            predictions[iter] = prediction
            iter += 1
        
        return predictions 

    def _visualize_helper(self, tree, level):
        """
        Helper function for visualize a decision tree at a given level of recursion.
        """
        tab_level = "  " * level
        val = tree.value if tree.value is not None else 0
        print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name, val))

    def visualize(self, branch=None, level=0):
        
        if not branch:
            branch = self.tree
        self._visualize_helper(branch, level)

        for branch in branch.branches:
            self.visualize(branch, level+1)
        

def information_gain(features, attribute_index, targets):

    child_column = features[:, attribute_index]
    child_labels = np.unique(child_column)
    parent_labels = np.unique(targets)
    weights = {}
    prior = {}
    entropy_child = 0
    entropy_parent = 0

    for child in child_labels:
        weights[child] = np.count_nonzero(child_column == child)/child_column.size
        prior[child] = {}

        child_label_idxs = np.argwhere(child_column == child)
        child_parent = np.zeros((len(child_label_idxs)))

        iter = 0
        for i in child_label_idxs:
            child_parent[iter] = targets[i[0]]
            iter += 1

        for parent_label in parent_labels:
            prior[child][parent_label] = np.count_nonzero(child_parent == parent_label)/np.count_nonzero(child_column == child)

    for child in prior.keys():
        label_entropy = 0
        for repeat in prior[child].values():
            label_entropy -= 0 if repeat == 0 else repeat*np.log2(repeat)
        entropy_child += weights[child] * label_entropy

    for parent_label in parent_labels:
        repeat = np.count_nonzero(targets == parent_label)/targets.size
        entropy_parent -= repeat*np.log2(repeat)

    information_gain = entropy_parent - entropy_child

    return information_gain
#################################################################################################################################
fraction = 1.0
data_path = 'data/blobs.csv'
features, targets, attribute_names = load_data(data_path)
train_features, train_targets, test_features, test_targets = train_test_split(features, targets, fraction)


decision_tree = DecisionTree(attribute_names)
decision_tree.fit(train_features, train_targets)
decision_tree.visualize()
predictions = decision_tree.predict(test_features)
confusion_matrix = compute_confusion_matrix(test_targets, predictions)
accuracy = compute_accuracy(test_targets, predictions)
precision, recall = compute_precision_and_recall(test_targets, predictions)
f1_measure = compute_f1_measure(test_targets, predictions)

print(f"accuracy: {accuracy}")

# fraction = 1.0
# data_path = 'data/transform_me.csv'
# features, targets, attribute_names = load_data(data_path)
# train_features, train_targets, test_features, test_targets = train_test_split(features, targets, fraction)

# decision_tree = DecisionTree(attribute_names)
# decision_tree.fit(train_features, train_targets)
# plot_decision_regions(test_features, test_targets, decision_tree, 'Decision Tree Decision Regions for Transform Me')
# # plt.show()
# plt.savefig("decision_tree_plots/Decision_Tree_Decision_Regions_for_Transform_Me.png")