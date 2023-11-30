import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithLogits(nn.BCEWithLogitsLoss):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = super().forward(pred_real, torch.ones_like(pred_real))
            loss_fake = super().forward(pred_fake, torch.zeros_like(pred_fake))
            return loss_real + loss_fake
        else:
            loss = super().forward(pred_real, torch.ones_like(pred_real))
            return loss


class Hinge(nn.Module):
    def forward(self, pred_real, pred_fake=None, pred_blur=None):
        if pred_fake is not None and pred_blur is not None:
            loss_real = F.relu(1 - pred_real).mean()
            loss_fake = F.relu(1 + pred_fake).mean()
            loss_blur = F.relu(1 + pred_blur).mean()
            return loss_real + loss_fake + loss_blur
        else:
            loss = -pred_real.mean()
            return loss


class Hinge1(nn.Module):
    def forward(self, pred_real, pred_fake=None, pred_blur=None, pred_his=None):
        if pred_fake is not None and pred_blur is not None and pred_his is not None:
            loss_real = F.relu(1 - pred_real).mean()
            loss_fake = F.relu(1 + pred_fake).mean()
            loss_blur = F.relu(1 + pred_blur).mean()
            loss_his = F.relu(1 + pred_his).mean()
            return loss_real + loss_fake + loss_blur + loss_his

class Wasserstein(nn.Module):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = -pred_real.mean()
            loss_fake = pred_fake.mean()
            loss = loss_real + loss_fake
            return loss
        else:
            loss = -pred_real.mean()
            return loss


class Softplus(nn.Module):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = F.softplus(-pred_real).mean()
            loss_fake = F.softplus(pred_fake).mean()
            loss = loss_real + loss_fake
            return loss
        else:
            loss = F.softplus(-pred_real).mean()
            return loss
