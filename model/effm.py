import torch
from torch import nn
from torchvision.models import resnet34, ResNet34_Weights
import torch.nn.functional as F
from timm.models.helpers import named_apply
from functools import partial
# from pvt import pvt_v2_b0, pvt_v2_b2
from model.pvtmodify import pvt_v2_bb
import math



class COM(nn.Module):
    def __init__(self, channel):
        super(COM, self).__init__()

        act_fn = nn.ReLU(inplace=True)

        self.layer_10 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, groups=channel)
        self.layer_20 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, groups=channel)

        self.layer_11 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, groups=channel),
            nn.BatchNorm2d(channel),
            act_fn,
        )

        self.layer_21 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, groups=channel),
            nn.BatchNorm2d(channel),
            act_fn,
        )

        self.layer_ful1 = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            act_fn,
        )


        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.channel_mul_conv1 = nn.Sequential(
            nn.Conv2d(channel, channel // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 16, channel, kernel_size=1))
        
        self.channel_mul_conv2 = nn.Sequential(
            nn.Conv2d(channel, channel // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 16, channel, kernel_size=1))
        
    def forward(self, g, l):
        
        x_g = self.layer_10(g)
        x_l = self.layer_20(l)

        g_w = self.sigmoid(x_g)
        l_w = self.sigmoid(x_l)

        x_g_w = g.mul(l_w)
        x_l_w = l.mul(g_w)

        x_g_r = x_g_w + g
        x_l_r = x_l_w + l

        x_g_r = self.layer_11(x_g_r)
        x_l_r = self.layer_21(x_l_r)

        x_g_r = x_g_r * torch.sigmoid(self.channel_mul_conv1(x_g_r))
        x_l_r = x_l_r * torch.sigmoid(self.channel_mul_conv2(x_l_r))
        ful_out = torch.cat((x_g_r, x_l_r), dim=1)

        avgout = torch.mean(ful_out, dim=1, keepdim=True)
        maxout, _ = torch.max(ful_out, dim=1, keepdim=True)
        mask = self.conv2d(torch.cat([avgout, maxout], dim=1))
        mask = self.sigmoid(mask)

        out1 = self.layer_ful1(ful_out) * mask

        return out1


class Conv_Bn_Relu(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size, stride=1, dilation=1, groups=1, bias=False, skip=False,
                 inplace=True):
        super(Conv_Bn_Relu, self).__init__()
        self.has_skip = skip and dim_in == dim_out
        padding = math.ceil((kernel_size - stride) / 2)
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias)
        self.norm = nn.BatchNorm2d(dim_out)
        self.act = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class MSFGM(nn.Module):
    def __init__(self, channel):
        super(MSFGM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 320, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out)


class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
    

class MRFFM(nn.Module):
    def __init__(self, in_channel, out_channel, exp_ratio=1.0):
        super(MRFFM, self).__init__()

        mid_channel = in_channel * exp_ratio

        self.DWConv = Conv_Bn_Relu(mid_channel, mid_channel, kernel_size=3, groups=out_channel // 2)
        self.DWConv3x3 = Conv_Bn_Relu(in_channel // 8, in_channel // 8, kernel_size=3, groups=in_channel // 8)
        self.DWConv5x5 = Conv_Bn_Relu(in_channel // 8, in_channel // 8, kernel_size=5, groups=in_channel // 8)
        self.DWConv7x7 = Conv_Bn_Relu(in_channel // 8, in_channel // 8, kernel_size=7, groups=in_channel // 8)
        self.PWConv1 = Conv_Bn_Relu(in_channel, mid_channel, kernel_size=1)
        self.PWConv2 = Conv_Bn_Relu(mid_channel, out_channel, kernel_size=1)
        self.norm = nn.BatchNorm2d(in_channel)
        self.max1 = nn.MaxPool2d(3, stride=1, padding=1)
        self.max2 = nn.MaxPool2d(5, stride=1, padding=2)
        self.avg1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.avg2 = nn.AvgPool2d(kernel_size=3, stride=3, padding=0)
        self.avg3 = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        channels = x.size(1)
        channels_per_part = channels // 8

        x1 = x[:, :channels_per_part, :, :]
        # print(x1.shape)
        x2 = x[:, channels_per_part:2 * channels_per_part, :, :]
        # print(x2.shape)
        x3 = x[:, 2 * channels_per_part:3 * channels_per_part, :, :]
        # print(x3.shape)
        x4 = x[:, 3 * channels_per_part:4 * channels_per_part, :, :]
        # print(x4.shape)
        x5 = x[:, 4 * channels_per_part:5 * channels_per_part, :, :]
        # print(x5.shape)
        x6 = x[:, 5 * channels_per_part:6 * channels_per_part, :, :]
        # print(x6.shape)
        x7 = x[:, 6 * channels_per_part:7 * channels_per_part, :, :]
        # print(x7.shape)
        x8 = x[:, 7 * channels_per_part:, :, :]
        # print(x8.shape)


        x1 = self.max1(x1)
        x2 = self.max2(x2)

        x3 = self.DWConv3x3(x3)
        x4 = self.DWConv5x5(x4)
        x5 = self.DWConv7x7(x5)
 
        x6 = self.avg1(x6)
        x6 = nn.functional.interpolate(x6, size=shortcut.size()[2:], mode='bilinear', align_corners=False)
        x7 = self.avg2(x7)
        x7 = nn.functional.interpolate(x7, size=shortcut.size()[2:], mode='bilinear', align_corners=False)
        x8 = self.avg3(x8)
        x8 = nn.functional.interpolate(x8, size=shortcut.size()[2:], mode='bilinear', align_corners=False)

        x2 = x1 * x2
        x3 = x2 * x3
        x4 = x3 * x4
        x5 = x4 * x5
        x6 = x5 * x6
        x7 = x6 * x7
        x8 = x7 * x8

        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=1)
        x = self.PWConv1(x)
        x = x + self.DWConv(x)
        x = self.PWConv2(x)
        x = x + shortcut

        return x
    

class Masnet(nn.Module):
    def __init__(self, num_classes=9):
        super(Masnet, self).__init__()

        self.resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1, progress=True)

        self.backbone = pvt_v2_bb()

        # self.backbone = pvt_v2_b2() 
        # path = r'E:\Code\renew\pvt_v2_b2.pth'
        # save_model = torch.load(path)
        # model_dict = self.backbone.state_dict()
        # state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        # model_dict.update(state_dict)
        # self.backbone.load_state_dict(model_dict)

        self.conv1 = BasicConv2d(128, 64, 1)
        self.conv2 = BasicConv2d(256, 128, 1)
        self.conv3 = BasicConv2d(512, 320, 1)

        self.ff1 = COM(channel=64)
        self.ff2 = COM(channel=128)
        self.ff3 = COM(channel=320)
        
        self.cs1 = RFB_modified(64, 32)
        self.cs2 = RFB_modified(128, 32)
        self.cs3 = RFB_modified(320, 32)

        self.agg = MSFGM(32)

        self.CA3 = CAB(320)
        self.CA2 = CAB(128)
        self.CA1 = CAB(64)

        self.SA = SAB()

        self.maf1 = MRFFM(in_channel=320, out_channel=320, exp_ratio=2)
        self.maf2 = MRFFM(in_channel=128, out_channel=128, exp_ratio=2)
        self.maf3 = MRFFM(in_channel=64, out_channel=64, exp_ratio=2)

        self.down = nn.MaxPool2d(4)
        self.up2 = nn.ConvTranspose2d(320, 128, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.ag3 = COM(channel=320)
        self.ag2 = COM(channel=128)
        self.ag1 = COM(channel=64)


        self.out_head4 = nn.Conv2d(320, num_classes, 1)
        self.out_head3 = nn.Conv2d(320, num_classes, 1)
        self.out_head2 = nn.Conv2d(128, num_classes, 1)
        self.out_head1 = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # # pvt

        p1, p2, p3 = self.backbone(x)

        # # resnet

        r = self.resnet.conv1(x)
        r = self.resnet.bn1(r)
        r = self.resnet.relu(r)

        r = self.resnet.layer1(r)

        r1 = self.resnet.layer2(r)
        r2 = self.resnet.layer3(r1)
        r3 = self.resnet.layer4(r2)

        r1 = self.conv1(r1)
        r2 = self.conv2(r2)
        r3 = self.conv3(r3)

        all1 = self.ff1(p1, r1)
        all2 = self.ff2(p2, r2)
        all3 = self.ff3(p3, r3)

        c1 = self.cs1(all1)
        c2 = self.cs2(all2)
        c3 = self.cs3(all3)

        out4 = self.agg(c3, c2, c1)
        o4 = self.down(out4)
        out4 = self.out_head4(out4)
        out4 = F.interpolate(out4, scale_factor=4, mode='bilinear')

        dd3 = self.ag3(o4, all3)
        dd3 = dd3 + o4
        d3 = self.CA3(dd3) * dd3
        d3 = self.SA(d3) * d3
        d3 = self.maf1(d3)

        out3 = self.out_head3(d3)
        out3 = F.interpolate(out3, scale_factor=16, mode='bilinear')

        d2 = self.up2(d3)
        x2 = self.ag2(d2, all2)
        d2 = d2 + x2

        d2 = self.CA2(d2) * d2
        d2 = self.SA(d2) * d2
        d2 = self.maf2(d2)

        out2 = self.out_head2(d2)
        out2 = F.interpolate(out2, scale_factor=8, mode='bilinear')

        d1 = self.up1(d2)
        x1 = self.ag1(d1, all1)
        d1 = d1 + x1

        d1 = self.CA1(d1) * d1
        d1 = self.SA(d1) * d1
        d1 = self.maf3(d1)

        out1 = self.out_head1(d1)
        out1 = F.interpolate(out1, scale_factor=4, mode='bilinear')

        return out4, out3, out2, out1


if __name__ == '__main__':
    x = torch.rand(5, 3, 224, 224)
    model = Masnet()
    y = model(x)
    print(y.shape)








