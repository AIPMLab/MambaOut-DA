import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def visda_acc(predict, all_label):
    matrix = confusion_matrix(all_label, predict)
    # ConfusionMatrixDisplay(matrix).plot()
    # plt.show()
    acc = matrix.diagonal()/matrix.sum(axis=1) * 100
    aacc = acc.mean()
    aa = [str(np.round(i, 2)) for i in acc]
    acc = ' '.join(aa)
    return aacc, acc