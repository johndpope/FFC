import torch
import torch.nn as nn

class FFCSE_block(nn.Module):
    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, groups=channels, bias=True),
            nn.Conv2d(channels, channels // r, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv_a2l = None if in_cl == 0 else nn.Sequential(
            nn.Conv2d(channels // r, channels // r, kernel_size=1, stride=1, padding=0, groups=channels // r, bias=True),
            nn.Conv2d(channels // r, in_cl, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.conv_a2g = None if in_cg == 0 else nn.Sequential(
            nn.Conv2d(channels // r, channels // r, kernel_size=1, stride=1, padding=0, groups=channels // r, bias=True),
            nn.Conv2d(channels // r, in_cg, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.conv1(x)

        x_l = 0 if self.conv_a2l is None else id_l * self.conv_a2l(x)
        x_g = 0 if self.conv_a2g is None else id_g * self.conv_a2g(x)

        return x_l, x_g

class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, padding=0, groups=in_channels * 2, bias=False),
            nn.Conv2d(in_channels * 2, out_channels * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()

        ffted = torch.rfft(x, signal_ndim=2, normalized=True)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()

        output = torch.irfft(ffted, signal_ndim=2,
                             signal_sizes=r_size[2:], normalized=True)

        return output


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=1, stride=1, padding=0, groups=out_channels // 2, bias=False),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s_h = h // split_no
            split_s_w = w // split_no
            xs = torch.cat(torch.split(x[:, :c // 4], split_s_h, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output

class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        # Depthwise Separable Convolution
        self.convl2l = nn.Sequential(
            nn.Conv2d(in_cl, in_cl, kernel_size, stride, padding, dilation, groups=in_cl, bias=bias),
            nn.Conv2d(in_cl, out_cl, kernel_size=1, stride=1, padding=0, bias=bias)
        ) if in_cl != 0 and out_cl != 0 else nn.Identity()

        self.convl2g = nn.Sequential(
            nn.Conv2d(in_cl, in_cl, kernel_size, stride, padding, dilation, groups=in_cl, bias=bias),
            nn.Conv2d(in_cl, out_cg, kernel_size=1, stride=1, padding=0, bias=bias)
        ) if in_cl != 0 and out_cg != 0 else nn.Identity()

        self.convg2l = nn.Sequential(
            nn.Conv2d(in_cg, in_cg, kernel_size, stride, padding, dilation, groups=in_cg, bias=bias),
            nn.Conv2d(in_cg, out_cl, kernel_size=1, stride=1, padding=0, bias=bias)
        ) if in_cg != 0 and out_cl != 0 else nn.Identity()

        self.convg2g = SpectralTransform(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu
        ) if in_cg != 0 and out_cg != 0 else nn.Identity()

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)

        return out_xl, out_xg


class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        # Depthwise Separable Convolution
        self.convl2l = nn.Sequential(
            nn.Conv2d(in_cl, in_cl, kernel_size, stride, padding, dilation, groups=in_cl, bias=bias),
            nn.Conv2d(in_cl, out_cl, kernel_size=1, stride=1, padding=0, bias=bias)
        ) if in_cl != 0 and out_cl != 0 else nn.Identity()

        self.convl2g = nn.Sequential(
            nn.Conv2d(in_cl, in_cl, kernel_size, stride, padding, dilation, groups=in_cl, bias=bias),
            nn.Conv2d(in_cl, out_cg, kernel_size=1, stride=1, padding=0, bias=bias)
        ) if in_cl != 0 and out_cg != 0 else nn.Identity()

        self.convg2l = nn.Sequential(
            nn.Conv2d(in_cg, in_cg, kernel_size, stride, padding, dilation, groups=in_cg, bias=bias),
            nn.Conv2d(in_cg, out_cl, kernel_size=1, stride=1, padding=0, bias=bias)
        ) if in_cg != 0 and out_cl != 0 else nn.Identity()

        self.convg2g = SpectralTransform(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu
        ) if in_cg != 0 and out_cg != 0 else nn.Identity()

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)

        return out_xl, out_xg
