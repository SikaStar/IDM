import torch
import torch.nn as nn


# Domain-specific BatchNorm
class DSBN2d(nn.Module):
    def __init__(self, planes):
        super(DSBN2d, self).__init__()
        self.num_features = planes
        self.BN_S = nn.BatchNorm2d(planes)
        self.BN_T = nn.BatchNorm2d(planes)

    def forward(self, x):
        if (not self.training):
            return self.BN_T(x)

        bs = x.size(0)
        assert (bs%2==0)
        split = torch.split(x, int(bs/2), 0)
        out1 = self.BN_S(split[0].contiguous())
        out2 = self.BN_T(split[1].contiguous())
        out = torch.cat((out1, out2), 0)
        return out


class DSBN1d(nn.Module):
    def __init__(self, planes):
        super(DSBN1d, self).__init__()
        self.num_features = planes
        self.BN_S = nn.BatchNorm1d(planes)
        self.BN_T = nn.BatchNorm1d(planes)

    def forward(self, x):
        if (not self.training):
            return self.BN_T(x)

        bs = x.size(0)
        assert (bs%2==0)
        split = torch.split(x, int(bs/2), 0)
        out1 = self.BN_S(split[0].contiguous())
        out2 = self.BN_T(split[1].contiguous())
        out = torch.cat((out1, out2), 0)
        return out


class DSBN2d_idm(nn.Module):
    def __init__(self, planes):
        super(DSBN2d_idm, self).__init__()
        self.num_features = planes
        self.BN_S = nn.BatchNorm2d(planes)
        self.BN_T = nn.BatchNorm2d(planes)
        self.BN_mix = nn.BatchNorm2d(planes)

    def forward(self, x):
        if (not self.training):
            return self.BN_T(x)

        bs = x.size(0)
        assert (bs%3==0)
        split = torch.split(x, int(bs/3), 0)
        out1 = self.BN_S(split[0].contiguous())
        out2 = self.BN_T(split[1].contiguous())
        out3 = self.BN_mix(split[2].contiguous())
        out = torch.cat((out1, out2, out3), 0)
        return out


class DSBN1d_idm(nn.Module):
    def __init__(self, planes):
        super(DSBN1d_idm, self).__init__()
        self.num_features = planes
        self.BN_S = nn.BatchNorm1d(planes)
        self.BN_T = nn.BatchNorm1d(planes)
        self.BN_mix = nn.BatchNorm1d(planes)

    def forward(self, x):
        if (not self.training):
            return self.BN_T(x)

        bs = x.size(0)
        assert (bs%3==0)
        split = torch.split(x, int(bs/3), 0)
        out1 = self.BN_S(split[0].contiguous())
        out2 = self.BN_T(split[1].contiguous())
        out3 = self.BN_mix(split[2].contiguous())
        out = torch.cat((out1, out2, out3), 0)
        return out



def convert_dsbn_idm(model, mixup_bn_names, idm=False):

    for _, (child_name, child) in enumerate(model.named_children()):
        # print(child_name)
        idm_flag = idm
        assert(not next(model.parameters()).is_cuda)
        for name in mixup_bn_names:
            if name in child_name:
                idm_flag = True
        if isinstance(child, nn.BatchNorm2d) and not idm_flag:
            m = DSBN2d(child.num_features)
            m.BN_S.load_state_dict(child.state_dict())
            m.BN_T.load_state_dict(child.state_dict())
            setattr(model, child_name, m)
        elif isinstance(child, nn.BatchNorm2d) and idm_flag:
            m = DSBN2d_idm(child.num_features)
            m.BN_S.load_state_dict(child.state_dict())
            m.BN_T.load_state_dict(child.state_dict())
            m.BN_mix.load_state_dict(child.state_dict())
            setattr(model, child_name, m)
        elif isinstance(child, nn.BatchNorm1d) and not idm_flag:
            m = DSBN1d(child.num_features)
            m.BN_S.load_state_dict(child.state_dict())
            m.BN_T.load_state_dict(child.state_dict())
            setattr(model, child_name, m)
        elif isinstance(child, nn.BatchNorm1d) and idm_flag:
            m = DSBN1d_idm(child.num_features)
            m.BN_S.load_state_dict(child.state_dict())
            m.BN_T.load_state_dict(child.state_dict())
            m.BN_mix.load_state_dict(child.state_dict())
            setattr(model, child_name, m)
        else:
            convert_dsbn_idm(child, mixup_bn_names, idm=idm_flag)


def convert_bn_idm(model, use_target=True):
    for _, (child_name, child) in enumerate(model.named_children()):
        assert(not next(model.parameters()).is_cuda)
        if isinstance(child, DSBN2d):
            m = nn.BatchNorm2d(child.num_features)
            if use_target:
                m.load_state_dict(child.BN_T.state_dict())
            else:
                m.load_state_dict(child.BN_S.state_dict())
            setattr(model, child_name, m)
        elif isinstance(child, DSBN2d_idm):
            m = nn.BatchNorm2d(child.num_features)
            if use_target:
                m.load_state_dict(child.BN_T.state_dict())
            else:
                m.load_state_dict(child.BN_S.state_dict())
            setattr(model, child_name, m)
        elif isinstance(child, DSBN1d):
            m = nn.BatchNorm1d(child.num_features)
            if use_target:
                m.load_state_dict(child.BN_T.state_dict())
            else:
                m.load_state_dict(child.BN_S.state_dict())
            setattr(model, child_name, m)
        elif isinstance(child, DSBN1d_idm):
            m = nn.BatchNorm1d(child.num_features)
            if use_target:
                m.load_state_dict(child.BN_T.state_dict())
            else:
                m.load_state_dict(child.BN_S.state_dict())
            setattr(model, child_name, m)
        else:
            convert_bn_idm(child, use_target=use_target)

