import torch.nn as nn
import numpy as np
import  torch


## Loss function

def f1_loss(pred, target):
    pred = torch.sigmoid(pred)

    tp = torch.sum(pred * target)
    tn = torch.sum((1-pred) * (1-target))
    fp = torch.sum((1-pred) * target)
    fn = torch.sum(pred * (1 - target))

    eps = torch.from_numpy(np.asarray(torch.finfo(torch.float32).eps))
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)

    f1 = 2 * p * r / (p + r + eps)
    f1[torch.isnan(f1)] = 0
    f1[torch.isinf(f1)] = 0
    return 1 - f1


def smoothiouLoss( pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1

    intersection = pred * target
    loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() - intersection.sum() + smooth)

    loss = 1 - loss.mean()

    return loss


def siouLoss( pred, target):

    pred = torch.sigmoid(pred)
    smooth = 1

    intersection = pred * target
    softiou = (intersection.sum() + smooth) / (pred.sum() + target.sum() -intersection.sum() + smooth)

    gt_siou = (target.sum() + smooth) / ((target > 0).sum() + smooth)
    softiou /= gt_siou
    assert 0. <= softiou <= 1.

    loss = 1 - softiou.mean()
    
    return loss


def sflLoss(preds, targets, gamma=2.0):
    '''
    Args:
    preds: [n,class_num]
    targets: [n,class_num]
    '''
    preds = preds.sigmoid()
    pt = (1 - targets) * torch.clamp((1 - preds), max=1., min=1e-4).log() + targets * torch.clamp(preds, max=1., min=1e-4).log()
    # pt = torch.clamp(pt, max=1., min=1e-4)
    w = torch.pow(torch.abs(targets - preds), gamma)
    loss = - w * pt
    return loss.sum() / preds.shape[0]

