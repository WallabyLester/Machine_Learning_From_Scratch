import numpy as np

# Do not import sklearn!

def compute_confusion_matrix(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the confusion matrix. The confusion 
    matrix for a binary classifier would be a 2x2 matrix as follows:

    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]

    YOU DO NOT NEED TO IMPLEMENT CONFUSION MATRICES THAT ARE FOR MORE THAN TWO 
    CLASSES (binary).
    
    Compute and return the confusion matrix.

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        confusion_matrix (np.array): 2x2 confusion matrix between predicted and actual labels

    """

    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for i in range(len(predictions)):
        if predictions[i] == 1 and actual[i] == 1:
            true_positives += 1
        elif predictions[i] == 0 and actual[i] == 0:
            true_negatives += 1
        elif predictions[i] == 1 and actual[i] == 0:
            false_positives += 1
        else: 
            false_negatives += 1

    confusion_matrix = np.array([[true_negatives, false_positives], [false_negatives, true_positives]])

    return confusion_matrix

def compute_accuracy(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the accuracy:

    Hint: implement and use the compute_confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        accuracy (float): accuracy
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    matrix = compute_confusion_matrix(actual, predictions)
    true_positives = matrix[1, 1]
    true_negatives = matrix[0, 0]

    accuracy = (true_positives + true_negatives) / np.sum(matrix)

    return accuracy


def compute_precision_and_recall(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the precision and recall:

    https://en.wikipedia.org/wiki/Precision_and_recall
    
    You MUST account for edge cases in which precision or recall are undefined
    by returning np.nan in place of the corresponding value.

    Hint: implement and use the compute_confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        precision (float): precision
        recall (float): recall
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    matrix = compute_confusion_matrix(actual, predictions)
    true_positives = matrix[1, 1]
    false_positives = matrix[0, 1]
    false_negatives = matrix[1, 0]

    # if the denominator sum is 0 then pass nan
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else np.nan
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else np.nan

    return precision, recall

def compute_f1_measure(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the F1-measure:

    https://en.wikipedia.org/wiki/Precision_and_recall#F-measure
   
    Because the F1-measure is computed from the precision and recall scores, you
    MUST handle undefined (NaN) precision or recall by returning np.nan. You
    should also consider the case in which precision and recall are both zero.

    Hint: implement and use the compute_precision_and_recall function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        f1_measure (float): F1 measure of dataset (harmonic mean of precision and 
        recall)
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    # 2 * p*r/p+r
    precision, recall = compute_precision_and_recall(actual, predictions)

    f1_measure = 2 * (precision * recall)/(precision + recall) if precision + recall else np.nan

    return f1_measure
