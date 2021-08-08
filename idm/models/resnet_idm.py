from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
from collections import OrderedDict
from .idm_module import IDM


__all__ = ['ResNet', 'resnet18_idm', 'resnet34_idm', 'resnet50_idm',
           'resnet101_idm', 'resnet152_idm']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNet.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)

        self.conv = nn.Sequential(OrderedDict([
            ('conv1', resnet.conv1),
            ('bn1', resnet.bn1),
            ('relu', resnet.relu),
            ('maxpool', resnet.maxpool)]))

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.idm1 = IDM(channel=64)
        self.idm2 = IDM(channel=256)
        self.idm3 = IDM(channel=512)
        self.idm4 = IDM(channel=1024)
        self.idm5 = IDM(channel=2048)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        if not pretrained:
            self.reset_params()

    def forward(self, x, output_prob=False, stage=0):
        bs = x.size(0)
        # x = self.base(x)
        x = self.conv(x)
        if stage==0 and self.training:
            x, attention_lam = self.idm1(x)
        x = self.layer1(x)
        if stage==1 and self.training:
            x, attention_lam = self.idm2(x)
        x = self.layer2(x)
        if stage==2 and self.training:
            x, attention_lam = self.idm3(x)
        x = self.layer3(x)
        if stage==3 and self.training:
            x, attention_lam = self.idm4(x)
        x = self.layer4(x)
        if stage==4 and self.training:
            x, attention_lam = self.idm5(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        if (self.training is False and output_prob is False):
            bn_x = F.normalize(bn_x)
            return bn_x

        if self.norm:
            norm_bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
        else:
            return bn_x

        if self.norm:
            return prob, x, norm_bn_x
        
        else:
            return prob, x, attention_lam

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def resnet18_idm(**kwargs):
    return ResNet(18, **kwargs)


def resnet34_idm(**kwargs):
    return ResNet(34, **kwargs)


def resnet50_idm(**kwargs):
    return ResNet(50, **kwargs)


def resnet101_idm(**kwargs):
    return ResNet(101, **kwargs)


def resnet152_idm(**kwargs):
    return ResNet(152, **kwargs)

