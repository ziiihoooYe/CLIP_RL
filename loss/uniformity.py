import torch

def uniformity_loss(features, t=2.0, num_samples=100000):
    device = features.device
    N = features.size(0)

    idx1 = torch.randint(low=0, high=N, size=(num_samples,), device=device)
    idx2 = torch.randint(low=0, high=N, size=(num_samples,), device=device)

    diff = features[idx1] - features[idx2]
    dist2 = diff.pow(2).sum(dim=-1)

    values = torch.exp(-t * dist2)
    avg = values.mean()

    uniformity = torch.log(avg + 1e-8)

    return uniformity