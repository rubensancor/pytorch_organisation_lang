from sklearn.metrics import f1_score, precision_score, recall_score
import torch


def f1(preds, target):
    target = target.cpu().detach().numpy()
    preds = torch.max(preds, 1)[1].cpu().detach().numpy()
    return f1_score(target, preds, average='macro')


def prec(preds, target):
    target = target.cpu().detach().numpy()
    preds = torch.max(preds, 1)[1].cpu().detach().numpy()
    return precision_score(target, preds, average='macro')


def rec(preds, target):
    target = target.cpu().detach().numpy()
    preds = torch.max(preds, 1)[1].cpu().detach().numpy()
    return recall_score(target, preds, average='macro')
