# IEEE TCSVT paper - CSFM

from model import common
import torch.nn as nn
import torch


def make_model(args, parent=False):
    return CSFM(args)


# Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Spatial Attention (SA) Layer
class SALayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SALayer, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(channel, channel * reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel * reduction, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.convs(x)
        return x * y

# Channel-wise and spatial attention residual (CSAR) block
class CSAR(nn.Module):
    def __init__(self, n_feat):
        super(CSAR, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=True)
        )
        self.CA = CALayer(n_feat, reduction=16)
        self.SA = SALayer(n_feat, reduction=2)
        self.conv_1x1 = nn.Conv2d(2 * n_feat, n_feat, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        U = self.convs(x)
        U_CA = self.CA(U)
        U_SA = self.SA(U)
        y = self.conv_1x1(torch.cat([U_CA, U_SA], 1))
        return y + x


class CSAR_group(nn.Module):
    def __init__(self, n_feat, n_B=16):
        super(CSAR_group, self).__init__()
        modules_body = []
        for i in range(n_B):
            modules_body.append(CSAR(n_feat))

        self.CSAR_chain = nn.Sequential(*modules_body)

    def forward(self, x):
        y = self.CSAR_chain(x)
        return y


class CSFM(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(CSFM, self).__init__()
        n_feat = args.n_feats
        n_B = args.n_B
        n_M = args.n_M
        scale = args.scale[0]

        # RGB mean for DIV2K
        self.sub_mean = common.MeanShift(args.rgb_range)

        # define head module
        modules_head = [nn.Conv2d(args.n_colors, n_feat, kernel_size=3, padding=1, bias=True)]

        # define body
        self.conv_3x3 = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=True)
        self.fmm1 = CSAR_group(n_feat, n_B)
        self.node1 = nn.Conv2d(2 * n_feat, n_feat, kernel_size=1, padding=0, bias=True)
        self.fmm2 = CSAR_group(n_feat, n_B)
        self.node2 = nn.Conv2d(3 * n_feat, n_feat, kernel_size=1, padding=0, bias=True)
        self.fmm3 = CSAR_group(n_feat, n_B)
        self.node3 = nn.Conv2d(4 * n_feat, n_feat, kernel_size=1, padding=0, bias=True)
        self.fmm4 = CSAR_group(n_feat, n_B)
        self.node4 = nn.Conv2d(5 * n_feat, n_feat, kernel_size=1, padding=0, bias=True)
        self.fmm5 = CSAR_group(n_feat, n_B)
        self.node5 = nn.Conv2d(6 * n_feat, n_feat, kernel_size=1, padding=0, bias=True)
        self.fmm6 = CSAR_group(n_feat, n_B)
        self.node6 = nn.Conv2d(7 * n_feat, n_feat, kernel_size=1, padding=0, bias=True)
        self.fmm7 = CSAR_group(n_feat, n_B)
        self.node7 = nn.Conv2d(8 * n_feat, n_feat, kernel_size=1, padding=0, bias=True)
        self.fmm8 = CSAR_group(n_feat, n_B)
        self.node8 = nn.Conv2d(9 * n_feat, n_feat, kernel_size=1, padding=0, bias=True)
        self.conv_3x3_bu = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=True)

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feat, act=False),
            conv(n_feat, args.n_colors, 3)]

        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self.head = nn.Sequential(*modules_head)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        p_0 = self.conv_3x3(x)
        p_1 = self.node1(torch.cat([self.fmm1(p_0), p_0], 1))
        p_2 = self.node2(torch.cat([self.fmm2(p_1), p_1, p_0], 1))
        p_3 = self.node3(torch.cat([self.fmm3(p_2), p_2, p_1, p_0], 1))
        p_4 = self.node4(torch.cat([self.fmm4(p_3), p_3, p_2, p_1, p_0], 1))
        p_5 = self.node5(torch.cat([self.fmm5(p_4), p_4, p_3, p_2, p_1, p_0], 1))
        p_6 = self.node6(torch.cat([self.fmm6(p_5), p_5, p_4, p_3, p_2, p_1, p_0], 1))
        p_7 = self.node7(torch.cat([self.fmm7(p_6), p_6, p_5, p_4, p_3, p_2, p_1, p_0], 1))
        p_8 = self.node8(torch.cat([self.fmm8(p_7), p_7, p_6, p_5, p_4, p_3, p_2, p_1, p_0], 1))
        res = self.conv_3x3_bu(p_8)

        y = self.tail(res + x)
        y = self.add_mean(y)
        return y

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
