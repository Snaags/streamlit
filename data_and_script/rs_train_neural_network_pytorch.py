"""
A script for training and evaluating a neural network.
"""

import logging
import os
import sys
import time
import torch
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
import networks_pytorch
import feed_forward_model_builder

def save_history_plot(history, filename):
    """
    Saves a plot of neural network training history
    """
    plt.figure()
    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label='test')
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
    print(y)
    n_classes = len(np.unique(y))
    n_samples = X.shape[0]
    n_features = X.shape[1]

    logger.info(f"Number of classes: {n_classes}")
    logger.info(f"Number of samples: {n_samples}")
    logger.info(f"Number of features: {n_features}")

    # create and fit scaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = torch.Tensor(X)
    y = torch.Tensor(y)
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
    val_loss_list = []
    loss_list = []
    index = 0
    for train, test in get_folds(X, y, groups=groups,
                                 n_folds=NUM_VALIDATION_FOLDS):

        val_loss_list = []
        loss_list = []
        model,criterion ,optimizer = networks_pytorch.build_sequential(n_features, n_classes, n_neurons_l1,
                                        n_neurons_l2, layers, learning_rate,
                                        0.0, l1_dropout, l2_dropout, l1_reg,
                                        l2_reg,batch_size)



        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for inputs,labels in zip(X[train], y[train]):

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)

                outputs = outputs.float()
                labels = labels.float()
                labels = labels.unsqueeze(-1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()   # print every 2000 mini-batches
            history =pd.DataFrame()
            loss_list.append(running_loss)
            history["loss"] = loss_list




            validation_loss = 0

            for inputs,labels in zip(X[test], y[test]):

                # zero the parameter gradients

                # forward + backward + optimize
                outputs = model(inputs)

                outputs = outputs.float()
                labels = labels.float()
                labels = labels.unsqueeze(-1)

                loss = criterion(outputs, labels)

                # print statistics
                validation_loss += loss.item()   # print every 2000 mini-batches
            val_loss_list.append(validation_loss)
            history["val_loss"] = val_loss_list





        plot_file = os.path.join(output_path, f"cv_{index}")
        save_history_plot(history, plot_file)

        """
        Get prediction scores. For binary classification, the predicted class
        is 0 when the score is below the prediction threshold, and 1 when
        above it. When there are multiple classes, the predicted class is
        that which has the highest score.
        """
        predicted_scores = []
        for i in X[test]:
            predicted_scores.append(model(i).detach().numpy())
        predicted_scores = np.asarray(predicted_scores)
        if n_classes == 2:
            y_pred = np.where(predicted_scores > PREDICTION_THRESHOLD, 1, 0)
            y_pred = np.ravel(y_pred)
        else:
            y_pred = np.argmax(predicted_scores, axis=1)

        test_labels = y[test].detach().numpy()

        """
        # Making the Confusion Matrix
        cm = sklearn.metrics.confusion_matrix(test_labels, Y_pred_thres)

        recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) * 100
        acc = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + cm[0, 1]
            + cm[1, 0]) * 100

        accscores_50.append(acc)
        recallscores_50.append(recall)

        fpr, tpr, _ = sklearn.metrics.roc_curve(test_labels, Y_pred)
        fprs.append(fpr)
        tprs.append(tpr)
        """

        # Plot the confusion matrix for the current fold
        conf_mx = evaluation.ConfusionMatrixModel(test_labels, y_pred)
        evaluation.confusion_matrix_plot(conf_mx)
        plt.savefig(os.path.join(output_path, f'confusion_matrix_{index}'))

        # Log and store the accuracy for the current fold
        # with the nominal threshold
        n_correct = np.count_nonzero(y_pred == test_labels)
        print("Correct Answers: ",n_correct)
        acc_fold = 100 * n_correct / float(len(y_pred))
        logger.info(f"Fold {index} accuracy: {acc_fold}")
        accscores.append(acc_fold)

        # Additional output for binary classification
        if n_classes == 2:
            cm = sklearn.metrics.confusion_matrix(test_labels, y_pred)
            recallscores.append(cm[1, 1] / (cm[1, 1] + cm[1, 0]) * 100)

            fpr, tpr, _ = sklearn.metrics.roc_curve(test_labels, y_pred)
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

    model, optimizer = feed_forward_model_builder.init_system(hyperparameters, n_features,n_classes)

    m_old,criterion ,opt = networks_pytorch.build_sequential(n_features, n_classes, n_neurons_l1,
                                    n_neurons_l2, layers, learning_rate,
                                    0.0, l1_dropout, l2_dropout, l1_reg,
                                    l2_reg,batch_size)



    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for inputs,labels in zip(X, y):

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            outputs = outputs.float()
            labels = labels.float()
            labels = labels.unsqueeze(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()   # print every 2000 mini-batches



    print('Execution time: ', (time.time() - start_time), ' seconds')


if __name__ == '__main__':
    main()
