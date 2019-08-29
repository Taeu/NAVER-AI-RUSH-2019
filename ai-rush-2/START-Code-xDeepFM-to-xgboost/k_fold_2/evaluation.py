# this is for reference only. you don't need to use this file.
import argparse

import numpy as np
# from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings("ignore")


def evaluate(y_true, y_pred):
    """
    Args:
      y_true (numpy array): ground truth class labels
      y_pred (numpy array): predicted class labels
    Returns:
      score (numpy float): Mean Squared Error.
    """
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    score = f1_score(y_true=y_true, y_pred=y_pred, pos_label=1)
    return score.item()


def evaluation_metrics(prediction_file, groundtruth_file):
    """
      Args:
        prediction_file (str): path to the file that stores the predicted labels
        groundtruth_file (str): path to the file that stores the ground truth labels
      Returns:
        acc: float top-1 accuracy.
    """
    y_pred = np.loadtxt(prediction_file)

    with open(groundtruth_file, 'r') as f:
        next(f)
        lines = f.readlines()

    y_true = []  # pctr e.g. [0.2 0.3 ... 0.001] delimiter == ' '
    for line in lines:
        line = line.rstrip('\n')  # in case of linefeeder
        line = int(line)
        y_true.append(line)

    y_true = np.array(y_true)

    return evaluate(y_true, y_pred)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--prediction', type=str, default='pred_example.txt')
    config = args.parse_args()
    test_label_path = '/data/airush2/test/test_label'
    try:
        print(evaluation_metrics(config.prediction, test_label_path))
    except:
        print("-1")
