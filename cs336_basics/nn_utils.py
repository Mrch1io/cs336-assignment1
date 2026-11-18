import torch


def softmax(x, dim):
    x = x - x.max(dim=dim, keepdim=True).values
    exp = torch.exp(x)
    return exp / exp.sum(dim=dim, keepdim=True)


def cross_entropy(inputs, targets):
    log_probs = inputs - torch.logsumexp(inputs, dim=-1, keepdim=True)
    return -log_probs.gather(-1, targets.unsqueeze(-1)).mean()


def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.sqrt(sum((g**2).sum() for g in grads))
    if total_norm > max_l2_norm:
        for g in grads:
            g.mul_(max_l2_norm / (total_norm + eps))
