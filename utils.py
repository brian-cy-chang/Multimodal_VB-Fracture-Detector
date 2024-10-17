import os

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable

def create_dir(save_dir):
    """
    Creates new directory if it does not exists

    Parameters
    ----------
    save_dir (str): desired directory name
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

def accuracy_fn(tp, tn, total):
    """
    Calculate accuracy by comparing model predictions to ground truth labels

    Parameters
    ----------
    tp (int): true positives per epoch
    tn (int): true negatives per epoch
    total (int): number of data points per epoch
    """
    acc = ((tp+tn) / total) * 100 
    return acc

class WeightedFocalLoss(nn.Module):
    """
    Non weighted version of Focal Loss

    from https://web.archive.org/web/20221101163651/https://amaarora.github.io/2020/06/29/FocalLoss.html
    """
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha])
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Move alpha to the same device as inputs
        if inputs.is_cuda:
            self.alpha = self.alpha.cuda()

        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

class SaveBestModel_F1:
    """
    Saves best model based on validation F1 score to .pth file
    """
    def __init__(self, save_path):
        self.best_f1 = 0.0
        self.epoch = 0
        self.save_path = save_path

    def __call__(self, model, optimizer, criterion, epoch, f1):
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.epoch = epoch
            print(f"Saving model at epoch {epoch+1} with validation F1 score = {f1:.4f}")
            print(f"\n")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'f1_score': f1
            }, self.save_path + ".pth")

class SaveBestModel_ValidationLoss:
    """
    Saves best model based on validation loss to .pth file
    """
    def __init__(self, save_path):
        self.best_val_loss = float('inf')
        self.epoch = 0
        self.save_path = save_path

    def __call__(self, model, optimizer, criterion, epoch, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epoch = epoch
            print(f"Saving model at epoch {epoch+1} with validation loss = {val_loss:.4f}")
            print(f"\n")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'val_loss': val_loss
                }, self.save_path + ".pth")

# def sigmoid_focal_loss(
#     inputs: torch.Tensor,
#     targets: torch.Tensor,
#     alpha: float = -1,
#     gamma: float = 2,
#     reduction: str = "none",
# ) -> torch.Tensor:
#     """
#     From https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py

#     Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

#     Parameters
#     ----------
#     inputs: A float tensor of arbitrary shape.
#             The predictions for each example.
#     targets: A float tensor with the same shape as inputs. Stores the binary
#             classification label for each element in inputs
#             (0 for the negative class and 1 for the positive class).
#     alpha: (optional) Weighting factor in range (0,1) to balance
#             positive vs negative examples. Default = -1 (no weighting).
#     gamma: Exponent of the modulating factor (1 - p_t) to
#             balance easy vs hard examples.
#     reduction: 'none' | 'mean' | 'sum'
#             'none': No reduction will be applied to the output.
#             'mean': The output will be averaged.
#             'sum': The output will be summed.

#     Returns
#     -------
#     Loss tensor with the reduction option applied.
#     """
#     inputs = inputs.float()
#     targets = targets.float()
#     p = torch.sigmoid(inputs)
#     ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
#     p_t = p * targets + (1 - p) * (1 - targets)
#     loss = ce_loss * ((1 - p_t) ** gamma)

#     if alpha >= 0:
#         alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
#         loss = alpha_t * loss

#     if reduction == "mean":
#         loss = loss.mean()
#     elif reduction == "sum":
#         loss = loss.sum()

#     return loss


