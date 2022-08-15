import torch
import torch.nn as nn

from nets.CSPdarknet53_tiny import darknet53_tiny
from nets.attention import cbam_block, eca_block, se_block, SpatialAttention, ChannelAttention

attention_block = [se_block, cbam_block, eca_block, SpatialAttention, ChannelAttention]

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
def conv_dw(inp, oup, stride = 1):
    return nn.Sequential(
        # part1
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),

        # part2
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x


def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv_dw(in_filters, filters_list[0]),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m

class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi=3):
        super(YoloBody, self).__init__()
        self.phi            = phi
        self.backbone       = darknet53_tiny(None)

        self.conv_for_P5    = BasicConv(512,256,1)
        self.yolo_headP5    = yolo_head([512, len(anchors_mask[0]) * (5 + num_classes)],256)

        self.upsample       = Upsample(256,128)
        self.upsample1       = Upsample(384,192)
        self.yolo_headP4    = yolo_head([256, len(anchors_mask[1]) * (5 + num_classes)],384)
        self.yolo_headP3    = yolo_head([128, len(anchors_mask[2]) * (5 + num_classes)],320)
        if 1 <= self.phi and self.phi <= 5:
            self.feat1_att      = attention_block[self.phi - 1](256)
            self.feat2_att      = attention_block[self.phi - 1](512)
            self.upsample_att   = attention_block[self.phi - 1](128)
            self.upsample_att1   = attention_block[self.phi - 1](192)

    def forward(self, x):

        feat1, feat2,feat3 = self.backbone(x)
        if 1 <= self.phi and self.phi <= 5:
            feat1 = self.feat1_att(feat1)
            feat2 = self.feat2_att(feat2)

        # 13,13,512 -> 13,13,256
        P5 = self.conv_for_P5(feat2)
        # 13,13,256 -> 13,13,512 -> 13,13,255
        out0 = self.yolo_headP5(P5)
        # 13,13,256 -> 13,13,128 -> 26,26,128
        P5_Upsample = self.upsample(P5)
        # 26,26,256 + 26,26,128 -> 26,26,384
        if 1 <= self.phi and self.phi <= 3:
            P5_Upsample = self.upsample_att(P5_Upsample)
        P4 = torch.cat([P5_Upsample,feat1],axis=1)

        # 26,26,384 -> 26,26,256 -> 26,26,255
        out1 = self.yolo_headP4(P4)
        # print(out1.size())

        # 26,26,384 -> 26,26,192 -> 52,52,192
        P4_Upsample = self.upsample1(P4)
        # 52,52,192 + 52,52,128 -> 52,52,320
        if 1 <= self.phi and self.phi <= 3:
            P4_Upsample = self.upsample_att1(P4_Upsample)
        P3 = torch.cat([P4_Upsample,feat3],axis=1)
        out2 = self.yolo_headP3(P3)
        # print(out2.size())
        return out0, out1,out2

