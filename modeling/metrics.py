import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_recall_curve,
    zero_one_loss,
    accuracy_score
)


def hamming_score(preds, targets):
    """Compute Hamming Score.

    This function computes the Hamming Score, a performance metric used for multi-label classification tasks.
    The Hamming Score measures the similarity between the predicted labels and the ground truth labels, where
    a higher score indicates better prediction accuracy.

    :param preds: The predicted labels.
    :type preds: numpy array
    :param targets: The ground truth labels.
    :type targets: numpy array
    :return: The computed Hamming Score.
    :rtype: float
    """

    preds = (preds > optimize_accuracy(preds, targets)).astype(float)
    return 1 - hamming_loss(targets, preds)


def zero_one_score(preds, targets):
    """
    Compute Zero-One Score.

    This function computes the Zero-One Score, a performance metric used for
    multi-label classification tasks. The Zero-One Score measures the similarity
    between the predicted labels and the ground truth labels, where a higher score
    indicates better prediction accuracy. The Zero-One Score ranges from 0 to 1, with 1 being a perfect match.

    :param preds: The predicted labels.
    :type preds: numpy array
    :param targets: The ground truth labels.
    :type targets: numpy array
    :return: The computed Zero-One Score.
    :rtype: float
    """

    preds = (preds > optimize_accuracy(preds, targets)).astype(float)
    return 1 - zero_one_loss(targets, preds, normalize=True)


def mean_f1_score(preds, targets):
    """Compute Mean F1 Score.

    This function computes the Mean F1 Score, a performance metric used for multi-label
    classification tasks. The Mean F1 Score measures the trade-off between precision and recall,
    where a higher score indicates better prediction accuracy. The Mean F1 Score ranges from
    0 to 1, with 1 being a perfect match.

    :param preds: The predicted labels.
    :type preds: numpy array
    :param targets: The ground truth labels.
    :type targets: numpy array
    :return: The computed Mean F1 Score.
    :rtype: float
    """

    preds = (preds > optimize_f1_score(preds, targets)).astype(float)
    return f1_score(targets, preds, average="samples", zero_division=0)


def per_instr_f1_score(preds, targets):
    """Compute Per-Instrument F1 Score.

    This function computes the F1 Score for each instrument separately in a multi-label
    classification task. The Per-Instrument F1 Score measures the prediction accuracy for
    each instrument class independently. The F1 Score is the harmonic mean of precision and recall,
    where a higher score indicates better prediction accuracy. The Per-Instrument F1 Score ranges
    from 0 to 1, with 1 being a perfect match.

    :param preds: The predicted labels.
    :type preds: numpy array
    :param targets: The ground truth labels.
    :type targets: numpy array
    :return: The computed Per-Instrument F1 Score.
    :rtype: numpy array
    """

    preds = (preds > optimize_f1_score(preds, targets)).astype(float)
    return f1_score(targets, preds, average=None, zero_division=0)


def mean_average_precision(preds, targets):
    """
    Compute mean Average Precision (mAP).

    This function computes the mean Average Precision (mAP), a performance metric used
    for multi-label classification tasks. The mAP measures the average precision across
    all classes, taking into account the precision-recall trade-off, where a higher score
    indicates better prediction accuracy.

    :param preds: The predicted probabilities or scores.
    :type preds: numpy array
    :param targets: The ground truth labels.
    :type targets: numpy array
    :return: The computed mAP score.
    :rtype: float
    """

    return average_precision_score(targets, preds, average="samples")


def optimize_f1_score(preds, targets):
    """
    Optimize Threshold.

    This function optimizes the threshold for binary classification based on the predicted probabilities
    and ground truth labels. It computes the precision, recall, and F1 Score for each class separately
    using the precision_recall_curve function from sklearn.metrics module. It then selects the threshold
    that maximizes the F1 Score for each class.

    :param preds: The predicted probabilities.
    :type preds: numpy array
    :param targets: The ground truth labels.
    :type targets: numpy array
    :return: The optimized thresholds for binary classification.
    :rtype: numpy array
    """

    label_thresholds = np.empty(preds.shape[1])

    for i in range(preds.shape[1]):
        precision, recall, thresholds = precision_recall_curve(targets[:, i], preds[:, i])
        fscore = (2 * precision * recall) / (precision + recall)
        ix = np.argmax(fscore)
        best_thresh = thresholds[ix]
        label_thresholds[i] = best_thresh
        
    return label_thresholds
        
def optimize_accuracy(preds, targets):

    # Vary the threshold for each label and calculate accuracy for each threshold
    thresholds = np.arange(0.001, 1, 0.001)
    best_thresholds = []
    for i in range(preds.shape[1]):
        accuracies = []
        for th in thresholds:
            y_pred = (preds[:, i] >= th).astype(int) # Convert probabilities to binary predictions using the threshold
            acc = accuracy_score(targets[:, i], y_pred)
            accuracies.append(acc)
        # Find the threshold that gives the highest accuracy for this label
        best_idx = np.argmax(accuracies)
        best_threshold = thresholds[best_idx]
        best_thresholds.append(best_threshold)
    
    return best_thresholds

        
