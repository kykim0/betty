import random
import numpy as np
import torch
from sklearn.metrics import f1_score


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, average="macro"):

    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    return {
        "acc": acc * 100.0,
        "f1": f1 * 100.0,
        "acc_and_f1": (acc + f1) / 2 * 100.0,
    }
