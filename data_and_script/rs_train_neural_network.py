"""
A script for training and evaluating a neural network.
"""

import logging
import os
import sys
import time

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection
from sklearn.utils import class_weight
import evaluation
import networks


def save_history_plot(history, filename):
    """
    Saves a plot of neural network training history
    """
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig(filename)


def save_accuracy_recall_plot(accuracy, recall, thresholds):
    """
    Saves a plot of accuracy and recall vs different thresholds
    """
    plt.figure()
    plt.plot(thresholds, accuracy)
    plt.plot(thresholds, recall)
    plt.savefig('temp.png')


def get_logger(log_file_name) -> logging.Logger:
    """Returns a logger that writes to the console and the named file"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Handler for writing output to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    # Handler for writing output to file
    fh = logging.FileHandler(log_file_name)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    return logger


def get_folds(X, y, groups=None, n_folds=5):
    """
    Returns a list of (test, train) folds.
    If the "groups" parameter is provided, a GroupKFold will be used,
    so the same group will never be represented in both a test and train fold.
    Otherwise, a StratifiedKFold will be used, which aims to have equal
    proportions of each label in each fold.
    """
    if groups is None:
        kfold = sklearn.model_selection.StratifiedKFold(
            n_splits=n_folds, shuffle=True)

        splits = kfold.split(X, y)
    else:
        kfold = sklearn.model_selection.GroupKFold(
            n_splits=n_folds)

        splits = kfold.split(X, y, groups=groups)

    return splits


def main():
    """Main method: Train and evaluate neural network"""

    ################################################
    # CONFIGURATION START
    ################################################

    start_time = time.time()

    NUM_VALIDATION_FOLDS = 5
    PREDICTION_THRESHOLD = 0.5

    ################################################
    # CONFIGURATION END
    ################################################

    output_path = ''  # Modify this line if you want the output files written somewhere else

    log_path = os.path.join(output_path, '.log')
    logger = get_logger(log_path)

    # Read command line parameters
    SELECTED_FEATURES_PATH = sys.argv[1]
    PARAMS_PATH = sys.argv[2]

    logger.info(f"Feature table path: {SELECTED_FEATURES_PATH}")
    logger.info(f"Parameter path: {PARAMS_PATH}")

    # Load the data matrix X and label vector y
    df = pd.read_csv(SELECTED_FEATURES_PATH, index_col=0)
    groups = np.array(df['group_id'])
    y = np.array(df['label'])
    metadataList = ['sample_id', 'group_id', 'subsys_id', 'work_order_id',
                    'start_ts', 'end_ts', 'label']
    all_features = df.drop(columns=metadataList, errors='ignore')
    X = np.array(all_features)
    
    n_classes = len(np.unique(y))
    n_samples = X.shape[0]
    n_features = X.shape[1]

    logger.info(f"Number of classes: {n_classes}")
    logger.info(f"Number of samples: {n_samples}")
    logger.info(f"Number of features: {n_features}")

    # create and fit scaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    with open(PARAMS_PATH, 'r') as f:
        params = json.load(f)

    # Model build parameters
    n_neurons_l1 = params['n_neurons_l1']
    n_neurons_l2 = params['n_neurons_l2']
    layers = params['layers']
    learning_rate = params['learning_rate']
    l1_dropout = params['l1_dropout']
    l2_dropout = params['l2_dropout']
    l1_reg = params['l1_reg']
    l2_reg = params['l2_reg']

    # Model fit parameters
    batch_size = params['batch_size']
    epochs = params['epochs']

    accscores = []
    recallscores = []
    fprs = []
    tprs = []

    index = 0
    for train, test in get_folds(X, y, groups=groups,
                                 n_folds=NUM_VALIDATION_FOLDS):
        clf = networks.build_sequential(n_features, n_classes, n_neurons_l1,
                                        n_neurons_l2, layers, learning_rate,
                                        0.0, l1_dropout, l2_dropout, l1_reg,
                                        l2_reg)

        class_weights = class_weight.compute_class_weight(
            'balanced', classes=np.unique(y[train]), y=y[train])

        # Fitting the ANN to the Training set
        history = clf.fit(X[train], y[train], class_weight=class_weights,
                          validation_data=(X[test], y[test]),
                          verbose=0, batch_size=batch_size, epochs=epochs)

        plot_file = os.path.join(output_path, f"cv_{index}")
        save_history_plot(history, plot_file)

        """
        Get prediction scores. For binary classification, the predicted class
        is 0 when the score is below the prediction threshold, and 1 when
        above it. When there are multiple classes, the predicted class is
        that which has the highest score.
        """
        predicted_scores = clf.predict(X[test])
        if n_classes == 2:
            y_pred = np.where(predicted_scores > PREDICTION_THRESHOLD, 1, 0)
            y_pred = np.ravel(y_pred)
        else:
            y_pred = np.argmax(predicted_scores, axis=1)

        """
        # Making the Confusion Matrix
        cm = sklearn.metrics.confusion_matrix(Y[test], Y_pred_thres)

        recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) * 100
        acc = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + cm[0, 1]
            + cm[1, 0]) * 100

        accscores_50.append(acc)
        recallscores_50.append(recall)

        fpr, tpr, _ = sklearn.metrics.roc_curve(Y[test], Y_pred)
        fprs.append(fpr)
        tprs.append(tpr)
        """

        # Plot the confusion matrix for the current fold
        conf_mx = evaluation.ConfusionMatrixModel(y[test], y_pred)
        evaluation.confusion_matrix_plot(conf_mx)
        plt.savefig(os.path.join(output_path, f'confusion_matrix_{index}'))

        # Log and store the accuracy for the current fold
        # with the nominal threshold
        n_correct = np.count_nonzero(y_pred == y[test])
        acc_fold = 100 * n_correct / float(len(y_pred))
        logger.info(f"Fold {index} accuracy: {acc_fold}")
        accscores.append(acc_fold)

        # Additional output for binary classification
        if n_classes == 2:
            cm = sklearn.metrics.confusion_matrix(y[test], y_pred)
            recallscores.append(cm[1, 1] / (cm[1, 1] + cm[1, 0]) * 100)

            fpr, tpr, _ = sklearn.metrics.roc_curve(y[test], y_pred)
            fprs.append(fpr)
            tprs.append(tpr)

        index = index + 1

    logger.info(f"Accuracy Ave = {np.mean(accscores)}")
    logger.info(f"Accuracy Std = {np.std(accscores)}")

    if n_classes == 2:
        logger.info(f"Recall Ave = {np.mean(recallscores)}")
        logger.info(f"Recall Std = {np.std(recallscores)}")
        roc_path = os.path.join(output_path, 'roc.png')
        evaluation.save_cross_validated_roc_curve(fprs, tprs, roc_path)

    # Now train and save production classifier based on ALL data
    # (train plus test)
    joblib.dump(scaler, os.path.join(output_path, 'scaler_ANN.h5'))
    production_clf = networks.build_sequential(n_features, n_classes,
                                               n_neurons_l1, n_neurons_l2,
                                               layers, learning_rate, 0.0,
                                               l1_dropout, l2_dropout,
                                               l1_reg, l2_reg)

    class_weights = class_weight.compute_class_weight('balanced',
                                                      classes=np.unique(y),
                                                      y=y)

    # Fitting the ANN to ALL data
    production_clf.fit(X, y, batch_size=batch_size, epochs=epochs,
                       class_weight=class_weights, verbose=0)

    production_clf.save(os.path.join(output_path, 'ANN_classifier.h5'))

    print('Execution time: ', (time.time() - start_time), ' seconds')


if __name__ == '__main__':
    main()
