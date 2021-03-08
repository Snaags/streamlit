"""
Contains methods for evaluating classifier performance.
"""

import pandas as pd
import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np
import json


class ConfusionMatrixModel():
    def __init__(self, y, y_pred):
         self.raw_matrix = sklearn.metrics.confusion_matrix(y, y_pred)
         self.matrix = ConfusionMatrixModel.normalise(self.raw_matrix)

    def normalise(matrix):
        """
        Assuming the classifier is reasonably good, the confusion matrix will have much larger values on the
        diagonal than on the off-diagonal. Since we're typically interested in the errors the classifier makes,
        we create a 'normalised' version, where the diagonal elements are set to zero. Additionally, we divide
        each value in the confusion matrix by the number of samples in the corresponding class, so we are comparing
        error rates instead of absolute numbers of errors (which would make abundant classes look unfairly bad).
        """
        row_sums = matrix.sum(axis=1, keepdims=True)
        norm_matrix = matrix / row_sums
        np.fill_diagonal(norm_matrix, 0)
        return norm_matrix


def confusion_matrix_plot(model: ConfusionMatrixModel):
    fig, ax = plt.subplots()
    ax.matshow(model.matrix, cmap=plt.cm.gray)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('Actual class')
    ax.xaxis.set_label_position('top')


class RocModel():
    def __init__(self, y_true, y_score):
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)
        self.fpr = fpr
        self.tpr = tpr
        self.thresholds = thresholds

    def auc(self):
        return sklearn.metrics.auc(self.fpr, self.tpr)


def roc_plot(model: RocModel):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=0.8)
    ax.plot(model.fpr, model.tpr, color='b', label=r'ROC (AUC = %0.2f)' % model.auc(), lw=2, alpha=0.8)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver operating characteristic (ROC)")
    ax.legend(loc="lower right")

    plt.xlabel('False Positive Rate (Fall-out)')
    plt.ylabel('True Positive Rate (Recall)')



def save_cross_validated_roc_curve(fprs, tprs, filename):
    if not len(fprs) == len(tprs):
        raise ValueError("fprs and tprs must have the same length")

    interp_tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for (fpr, tpr) in zip(fprs, tprs):
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)
        aucs.append(sklearn.metrics.auc(fpr, tpr))

    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    std_tpr = np.std(interp_tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    fig, ax = plt.subplots() 
    
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=0.8)

    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw = 2, alpha=0.8)

    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver operating characteristic (ROC)")
    ax.legend(loc="lower right")

    plt.xlabel('False Positive Rate (Fall-out)')
    plt.ylabel('True Positive Rate (Recall)')

    fig.savefig(filename)


