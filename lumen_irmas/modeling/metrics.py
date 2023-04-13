from sklearn.metrics import hamming_loss, zero_one_loss, f1_score, average_precision_score, precision_recall_curve
import numpy as np


def hamming_score(preds, targets):
    preds = (preds > optimize_threshold(preds, targets)).astype(float)
    return 1 - hamming_loss(targets, preds)


def zero_one_score(preds, targets):
    preds = (preds > optimize_threshold(preds, targets)).astype(float)
    return 1 - zero_one_loss(targets, preds, normalize=True)


def mean_f1_score(preds, targets):
    preds = (preds > optimize_threshold(preds, targets)).astype(float)
    return f1_score(targets, preds, average="samples", zero_division=0)


def per_instr_f1_score(preds, targets):
    preds = (preds > optimize_threshold(preds, targets)).astype(float)
    return f1_score(targets, preds, average=None, zero_division=0)


def mAP(preds, targets):
    return average_precision_score(targets, preds, average="samples")

def optimize_threshold(preds, targets):
    label_thresholds = np.empty(preds.shape[1])

    for i in range(preds.shape[1]):
        precision, recall, thresholds = precision_recall_curve(targets[:, i], preds[:, i])
        fscore = (2 * precision * recall) / (precision + recall)
        ix = np.argmax(fscore)
        best_thresh = thresholds[ix]
        label_thresholds[i] = best_thresh

    return label_thresholds
