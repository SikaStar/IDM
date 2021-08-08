from __future__ import absolute_import

from torch import nn
import torch


class IDM(nn.Module):
    def __init__(self, channel=64):
        super(IDM, self).__init__()
        self.channel = channel
        self.adaptiveFC1 = nn.Linear(2*channel, channel)
        self.adaptiveFC2 = nn.Linear(channel, int(channel/2))
        self.adaptiveFC3 = nn.Linear(int(channel/2), 2)
        self.softmax = nn.Softmax(dim=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):

        if (not self.training):
            return x

        bs = x.size(0)
        assert (bs%2==0)
        split = torch.split(x, int(bs/2), 0)
        x_s = split[0].contiguous() # [B, C, H, W]
        x_t = split[1].contiguous()

        x_embd_s = torch.cat((self.avg_pool(x_s.detach()).squeeze(), self.max_pool(x_s.detach()).squeeze()), 1)  # [B, 2*C]
        x_embd_t = torch.cat((self.avg_pool(x_t.detach()).squeeze(), self.max_pool(x_t.detach()).squeeze()), 1)

        x_embd_s, x_embd_t = self.adaptiveFC1(x_embd_s), self.adaptiveFC1(x_embd_t) # [B, C]
        x_embd = x_embd_s+x_embd_t
        x_embd = self.adaptiveFC2(x_embd)
        lam = self.adaptiveFC3(x_embd)
        lam = self.softmax(lam) # [B, 2]
        x_inter = lam[:, 0].reshape(-1,1,1,1)*x_s + lam[:, 1].reshape(-1,1,1,1)*x_t
        out = torch.cat((x_s, x_t, x_inter), 0)
        return out, lam
