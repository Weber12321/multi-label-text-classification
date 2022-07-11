import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def accuracy_thresh(y_pred, y_true, thresh=0.5, average='macro', digits=3):
    """
    :param y_pred: the sigmoid predictions of the model output, expect array
    :param y_true: the true labels array
    :param thresh: the threshold of multi-label classification, set default to 0.5
    :param average: the average method of metrics, set default to weighted
    :param digits: round to the number of decimals
    :return: dictionary of accuracy, precision, recall and f1_score
    """
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)

    a = accuracy_score(
        y_true=np.array(y_true.bool()),
        y_pred=np.array(y_pred > thresh)
    )

    p = precision_score(
        y_true=np.array(y_true.bool()),
        y_pred=np.array(y_pred > thresh),
        average=average
    )
    r = recall_score(
        y_true=np.array(y_true.bool()),
        y_pred=np.array(y_pred > thresh),
        average=average
    )
    f = f1_score(
        y_true=np.array(y_true.bool()),
        y_pred=np.array(y_pred > thresh),
        average=average
    )

    return {
        'accuracy': round(a * 100, digits),
        'precision': round(p * 100, digits),
        'recall': round(r * 100, digits),
        'f1': round(f * 100, digits)
    }
