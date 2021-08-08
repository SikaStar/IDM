# Reference: Wang et al. Cross-Batch Memory for Embedding Learning, in CVPR 2020.
# https://github.com/msight-tech/research-xbm

import torch
from torch import nn
import torch.nn.functional as F


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

class TripletLossXBM(nn.Module):
    def __init__(self, margin=0.3, norm=False):
        super(TripletLossXBM, self).__init__()
        self.margin = margin
        self.norm = norm
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs_col, targets_col, inputs_row, targets_row):

        n = inputs_col.size(0)
        if self.norm:
            inputs_col = F.normalize(inputs_col)
            inputs_row = F.normalize(inputs_row)

        dist = euclidean_dist(inputs_col, inputs_row)

        # split the positive and negative pairs
        pos_mask = targets_col.expand(
            targets_row.shape[0], n
        ).t() == targets_row.expand(n, targets_row.shape[0])
        neg_mask = ~pos_mask
        # For each anchor, find the hardest positive and negative
        dist_ap, dist_an = [], []

        for i in range(n):
            dist_ap.append(dist[i][pos_mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][neg_mask[i]].min().unsqueeze(0))

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss
