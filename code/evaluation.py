import numpy as np
from sklearn import metrics

def maxf1_oneClass(y_true, y_score):
    prec, rec, _ = metrics.precision_recall_curve(y_true, y_score)
    return np.nanmax(2*prec*rec/(prec+rec))

def maxf1(y_true, y_score):
    y_true = np.squeeze(np.array(y_true))
    y_score = np.squeeze(np.array(y_score))
    if y_true.ndim > 1:
        n_class = y_true.shape[1]        
        return np.mean([maxf1_oneClass(y_true[i], y_score[i]) for i in xrange(n_class)])
    return maxf1_oneClass(y_true, y_score)
    
def topKPrecision_oneClass(y_true, y_score, K=[10, 20, 30]):
    sorted_idx = np.argsort(y_score)[::-1]
    prec_K = []
    for k in K:
        y_true_k = y_true[sorted_idx][:k]
        prec_K.append(1.0 * np.sum(y_true_k) / k)
    return np.array(prec_K)
        
def topKPrecision(y_true, y_score, K=[10, 20, 30]):
    y_true = np.squeeze(np.array(y_true))
    y_score = np.squeeze(np.array(y_score))
    if y_true.ndim > 1:
        n_class = y_true.shape[1]
        return np.mean([topKPrecision_oneClass(y_true[i], y_score[i]) for i in xrange(n_class)])
    return topKPrecision_oneClass(y_true, y_score, K)