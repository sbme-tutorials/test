import torch
import torch.nn as nn
import torch.nn.functional as F

class KpLoss(nn.Module):
    """Simple keypoint heatmap MSE loss with optional mask."""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target, mask=None):
        """
        pred, target: tensors with shape (B,C,H,W) or (C,H,W)
        mask: optional tensor broadcastable to pred/target that weights pixels (1 = keep, 0 = ignore)
        """
        if mask is not None:
            diff = (pred - target) * mask
            loss = diff.pow(2)
            if self.reduction == 'mean':
                denom = mask.sum().clamp_min(1.0)
                return loss.sum() / denom
            return loss.sum()
        return F.mse_loss(pred, target, reduction=self.reduction)


class CLALoss(nn.Module):
    """Simple classification loss using BCEWithLogits."""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, pred_logits, targets):
        """
        pred_logits: logits (B, N) or (N,)
        targets: binary labels (same shape)
        """
        return self.loss_fn(pred_logits, targets.float())
