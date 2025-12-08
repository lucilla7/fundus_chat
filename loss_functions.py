import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, eps=1e-6):
        pred = torch.sigmoid(pred)
        num = 2 * (pred * target).sum()
        den = pred.sum() + target.sum() + eps
        return 1 - num / den