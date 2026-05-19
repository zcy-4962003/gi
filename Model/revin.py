"""
Reversible Instance Normalization (RevIN)
==========================================
Mitigates distribution shift in non-stationary industrial data by
per-sample normalization across the feature dimension, then reversibly
denormalizing the output to restore original scale.

Reference: Kim et al., "Reversible Instance Normalization for
Accurate Time-Series Forecasting against Distribution Shift", ICLR 2022.
"""
import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization.

    For single-timestep sintering data, normalizes across the feature
    dimension for each sample, removing per-sample distribution shifts
    caused by varying operating conditions.

    Args:
        num_features: Number of input features
        eps: Small constant for numerical stability
        affine: If True, use learnable gamma and beta parameters
    """

    def __init__(self, num_features, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        self.mean = None
        self.std = None

    def forward(self, x, mode='norm'):
        """
        Args:
            x: (batch_size, seq_len, num_features)
            mode: 'norm' for normalization, 'denorm' for denormalization
        Returns:
            Normalized or denormalized tensor of same shape
        """
        if mode == 'norm':
            self.mean = x.mean(dim=-1, keepdim=True)     # (B, S, 1)
            self.std = x.std(dim=-1, keepdim=True) + self.eps

            x_norm = (x - self.mean) / self.std

            if self.affine:
                x_norm = x_norm * self.gamma + self.beta

            return x_norm

        elif mode == 'denorm':
            if self.mean is None or self.std is None:
                raise RuntimeError(
                    "RevIN.forward(mode='norm') must be called before denorm."
                )

            if self.affine:
                gamma_pooled = self.gamma.mean()
                beta_pooled = self.beta.mean()
                x = (x - beta_pooled) / (gamma_pooled + self.eps)

            x = x * self.std.squeeze(-1) + self.mean.squeeze(-1)
            return x

        else:
            raise ValueError(
                f"Unknown mode: {mode}. Use 'norm' or 'denorm'."
            )
