import numpy as np
import sklearn.metrics as metrics

def compute_eer(labels, predictions):
    fpr, tpr, _ = metrics.roc_curve(labels, predictions, pos_label=1)

    # Finding the point closest to the equal error rate
    eer = 1
    for i in range(len(fpr)):
        if np.abs(fpr[i] - (1 - tpr[i])) < eer:
            eer = np.abs(fpr[i] - (1 - tpr[i]))

    return eer