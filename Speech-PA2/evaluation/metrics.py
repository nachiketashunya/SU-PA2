import numpy as np
import sklearn.metrics as metrics

def compute_eer(labels, predictions):
    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions)
    
    # Interpolate ROC curve to find threshold where FPR equals FNR
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.abs(fpr - fnr))]
    
    # Calculate EER directly from the intersecting point
    eer_index = np.where(thresholds == eer_threshold)
    eer = (fpr[eer_index] + (1 - tpr[eer_index])) / 2
    
    return eer[0]