import torch
import numpy as np


def _rff_features(x: torch.Tensor, omegas: torch.Tensor, bias: torch.Tensor):
    # x: [N, D], omegas: [D, R], bias: [R]
    proj = x @ omegas  # [N, R]
    proj = proj + bias
    features = torch.sqrt(torch.tensor(2.0, device=x.device)) * torch.cos(proj)
    return features


def cov_rff(x: torch.Tensor, rff_dim: int, sigma: float, batchsize: int = 256, normalize: bool = True):
    """
    Compute covariance approximation via Random Fourier Features for Gaussian kernel.
    Returns (covariance_matrix, omegas, features) where
    - covariance_matrix: [R, R]
    - omegas: [D, R]
    - features: [N, R]
    """
    device = x.device
    N, D = x.shape

    # Sample omegas ~ N(0, 1/sigma^2 I)
    scale = 1.0 / sigma
    omegas = torch.randn(D, rff_dim, device=device) * scale
    bias = 2 * np.pi * torch.rand(rff_dim, device=device)

    # Compute features in batches to save memory
    feats = torch.empty((N, rff_dim), device=device, dtype=torch.float32)
    for start in range(0, N, batchsize):
        end = min(start + batchsize, N)
        feats[start:end] = _rff_features(x[start:end], omegas, bias)

    if normalize:
        feats = feats / np.sqrt(rff_dim)

    # Covariance in feature space
    cov = (feats.T @ feats) / N
    return cov, omegas, feats
