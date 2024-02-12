import numpy as np
import pandas as pd


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    confusion_matrix = np.zeros((2, 2))
    for i in range(len(y_true)):
        if y_true[i] == '1' and y_pred[i] == '1':
            confusion_matrix[0, 0]+= 1
        elif y_true[i] == '0' and y_pred[i] == '1':
            confusion_matrix[1, 0]+= 1
        elif y_true[i] == '1' and y_pred[i] == '0':
            confusion_matrix[0, 1]+= 1
        elif y_true[i] == '0' and y_pred[i] == '0':
            confusion_matrix[1, 1]+= 1
        else:
            raise ValueError('Arrays should contain only ones and zeros!')
    precision = confusion_matrix[0, 0]/(confusion_matrix[0, 0] + confusion_matrix[1, 0])
    recall = confusion_matrix[0, 0]/(confusion_matrix[0, 0] + confusion_matrix[0, 1])
    f1 = 2 * precision * recall/(precision + recall)
    accuracy = (confusion_matrix[0, 0] + confusion_matrix[1, 1])/np.sum(confusion_matrix)
    return np.array([precision, recall, f1, accuracy])


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    confusion_matrix = pd.crosstab(y_true, y_pred)
    true_pos_and_neg_sum = sum([confusion_matrix[i][i] for i in range(len(confusion_matrix))])
    accuracy = true_pos_and_neg_sum/len(y_true)
    return accuracy
    


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    ss_res = np.square(y_true - y_pred).sum()
    ss_total = np.square(y_true - np.mean(y_pred)).sum()
    r2 = 1 - ss_res/ss_total
    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    mse = 1 / len(y_true) * np.square(y_true - y_pred).sum()
    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    mse = 1 / len(y_true) * np.abs(y_true - y_pred).sum()
    return mse
    
