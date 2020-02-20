from sklearn.metrics import (f1_score,
                             precision_score,
                             recall_score,
                             accuracy_score)
from sklearn.metrics import confusion_matrix
import torch
import numpy as np


def f1(preds, target):
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy()
    if torch.is_tensor(preds):
        preds = torch.max(preds, 1)[1].cpu().detach().numpy()
    return f1_score(target, preds, average='macro', zero_division=0)


def prec(preds, target):
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy()
    if torch.is_tensor(preds):
        preds = torch.max(preds, 1)[1].cpu().detach().numpy()
    return precision_score(target, preds, average='macro', zero_division=0)


def rec(preds, target):
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy()
    if torch.is_tensor(preds):
        preds = torch.max(preds, 1)[1].cpu().detach().numpy()
    return recall_score(target, preds, average='macro', zero_division=0)


def acc(preds, target):
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy()
    if torch.is_tensor(preds):
        preds = torch.max(preds, 1)[1].cpu().detach().numpy()
    return accuracy_score(target, preds)


def create_confusion_matrix(preds, target, orgs_labels, normalize='all'):
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy()
    if torch.is_tensor(preds):
        preds = torch.max(preds, 1)[1].cpu().detach().numpy()
    return confusion_matrix(target, preds,
                            labels=[0, 1, 2, 3, 4],
                            normalize=normalize)


def user_metric(preds, target):
    preds = torch.max(preds, 1)[1].cpu().detach().numpy()
    preds = np.argmax(np.bincount(preds))
    target = target.cpu().detach().numpy()
    target = np.argmax(np.bincount(target))

    return preds, target
