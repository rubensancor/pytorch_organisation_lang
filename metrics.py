from sklearn.metrics import (f1_score,
                             precision_score,
                             recall_score,
                             accuracy_score)
from sklearn.metrics import confusion_matrix
import torch


def f1(preds, target):
    target = target.cpu().detach().numpy()
    preds = torch.max(preds, 1)[1].cpu().detach().numpy()
    return f1_score(target, preds, average='macro', zero_division=0)


def prec(preds, target):
    target = target.cpu().detach().numpy()
    preds = torch.max(preds, 1)[1].cpu().detach().numpy()
    return precision_score(target, preds, average='macro', zero_division=0)


def rec(preds, target):
    target = target.cpu().detach().numpy()
    preds = torch.max(preds, 1)[1].cpu().detach().numpy()
    return recall_score(target, preds, average='macro', zero_division=0)


def acc(preds, target):
    target = target.cpu().detach().numpy()
    preds = torch.max(preds, 1)[1].cpu().detach().numpy()
    return accuracy_score(target, preds)


def create_confusion_matrix(preds, target, normalize='all'):
    target = target.cpu().detach().numpy()
    preds = torch.max(preds, 1)[1].cpu().detach().numpy()
    return confusion_matrix(target, preds, normalize=normalize)
