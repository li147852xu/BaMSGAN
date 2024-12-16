import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEWithLogits(nn.BCEWithLogitsLoss):
    """
    Binary Cross Entropy (BCE) with logits loss, extended for adversarial losses.
    """
    def forward(self, pred_real, pred_fake=None):
        """
        Forward function for BCE with logits.

        Args:
            pred_real (Tensor): Predictions for real data.
            pred_fake (Tensor, optional): Predictions for fake data. Defaults to None.

        Returns:
            Tensor: Combined loss for real and fake data if pred_fake is provided, otherwise only real loss.
        """
        if pred_fake is not None:
            loss_real = super().forward(pred_real, torch.ones_like(pred_real))
            loss_fake = super().forward(pred_fake, torch.zeros_like(pred_fake))
            return loss_real + loss_fake
        else:
            return super().forward(pred_real, torch.ones_like(pred_real))

class Hinge(nn.Module):
    """
    Hinge loss for adversarial training.
    """
    def forward(self, pred_real, pred_fake=None, pred_blur=None):
        """
        Forward function for Hinge loss.

        Args:
            pred_real (Tensor): Predictions for real data.
            pred_fake (Tensor, optional): Predictions for fake data. Defaults to None.
            pred_blur (Tensor, optional): Predictions for blurred data. Defaults to None.

        Returns:
            Tensor: Total loss combining real, fake, and blurred data losses if provided.
        """
        if pred_fake is not None and pred_blur is not None:
            loss_real = F.relu(1 - pred_real).mean()  # Real data loss
            loss_fake = F.relu(1 + pred_fake).mean()  # Fake data loss
            loss_blur = F.relu(1 + pred_blur).mean()  # Blurred data loss
            return loss_real + loss_fake + loss_blur
        else:
            return -pred_real.mean()  # Default loss when only real data is provided

class Hinge1(nn.Module):
    """
    Extended Hinge loss with additional support for historical data.
    """
    def forward(self, pred_real, pred_fake=None, pred_blur=None, pred_his=None):
        """
        Forward function for extended Hinge loss.

        Args:
            pred_real (Tensor): Predictions for real data.
            pred_fake (Tensor, optional): Predictions for fake data. Defaults to None.
            pred_blur (Tensor, optional): Predictions for blurred data. Defaults to None.
            pred_his (Tensor, optional): Predictions for historical data. Defaults to None.

        Returns:
            Tensor: Combined loss for real, fake, blurred, and historical data.
        """
        if pred_fake is not None and pred_blur is not None and pred_his is not None:
            loss_real = F.relu(1 - pred_real).mean()  # Real data loss
            loss_fake = F.relu(1 + pred_fake).mean()  # Fake data loss
            loss_blur = F.relu(1 + pred_blur).mean()  # Blurred data loss
            loss_his = F.relu(1 + pred_his).mean()  # Historical data loss
            return loss_real + loss_fake + loss_blur + loss_his


# Wasserstein 损失
class Wasserstein(nn.Module):
    """
    Wasserstein loss for adversarial training.
    """
    def forward(self, pred_real, pred_fake=None):
        """
        Forward function for Wasserstein loss.

        Args:
            pred_real (Tensor): Predictions for real data.
            pred_fake (Tensor, optional): Predictions for fake data. Defaults to None.

        Returns:
            Tensor: Combined loss for real and fake data if pred_fake is provided, otherwise only real loss.
        """
        if pred_fake is not None:
            loss_real = -pred_real.mean()  # Real data loss
            loss_fake = pred_fake.mean()  # Fake data loss
            return loss_real + loss_fake
        else:
            return -pred_real.mean()  # Default loss when only real data is provided


# Softplus 损失
class Softplus(nn.Module):
    """
    Softplus loss for adversarial training.
    """
    def forward(self, pred_real, pred_fake=None):
        """
        Forward function for Softplus loss.

        Args:
            pred_real (Tensor): Predictions for real data.
            pred_fake (Tensor, optional): Predictions for fake data. Defaults to None.

        Returns:
            Tensor: Combined loss for real and fake data if pred_fake is provided, otherwise only real loss.
        """
        if pred_fake is not None:
            loss_real = F.softplus(-pred_real).mean()  # Real data loss
            loss_fake = F.softplus(pred_fake).mean()  # Fake data loss
            return loss_real + loss_fake
        else:
            return F.softplus(-pred_real).mean()  # Default loss when only real data is provided