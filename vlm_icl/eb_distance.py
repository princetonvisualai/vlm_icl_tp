"""Distance metrics for comparing hidden embeddings."""

import torch
import torch.nn.functional as F


def cosine_divergence(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine divergence: 1 - cosine_similarity. Returns values in [0, 2]."""
    return 1 - F.cosine_similarity(a, b, dim=-1)


def l2_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Euclidean (L2) distance."""
    return torch.norm(a - b, p=2, dim=-1)


def hellinger_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hellinger distance. Inputs are softmaxed first."""
    p = F.softmax(a, dim=-1)
    q = F.softmax(b, dim=-1)
    return torch.norm(torch.sqrt(p) - torch.sqrt(q), p=2, dim=-1) / (2 ** 0.5)


def kl_divergence(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """KL divergence D(a || b). Inputs are softmaxed first. Asymmetric."""
    log_p = F.log_softmax(a, dim=-1)
    log_q = F.log_softmax(b, dim=-1)
    return F.kl_div(log_q, log_p, reduction="none", log_target=True).sum(dim=-1)
