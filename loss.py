import torch
from torch import nn


class DomainLoss(nn.Module):
    def __init__(self, weights=None, n_feature_maps=1):
        super(DomainLoss, self).__init__()
        self.weights = weights if weights else n_feature_maps*[1]

    def forward(self, maps_source, maps_dest):
        loss = torch.tensor(0, dtype=torch.float).cuda()
        for a1, a2, weight in zip(maps_source, maps_dest, self.weights):
            loss += weight*torch.mean((a1.mean(dim=0) - a2.mean(dim=0))**2)
        return loss