import math
import torch
import torch.nn as nn
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
class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resblock_body, self).__init__()
        self.out_channels = out_channels
        self.conv1 = conv_dw(in_channels, out_channels)
        self.conv2 = conv_dw(out_channels//2, out_channels//2)
        self.conv3 = conv_dw(out_channels//2, out_channels//2)
        self.conv4 = BasicConv(out_channels, out_channels, 1)
        self.maxpool = nn.MaxPool2d([2,2],[2,2])
    def forward(self, x):
        x = self.conv1(x)
        route = x
        c = self.out_channels
        x = torch.split(x, c//2, dim = 1)[1]
        x = self.conv2(x)
        route1 = x
        x = self.conv3(x)
        x = torch.cat([x,route1], dim = 1)
        x = self.conv4(x)
        feat = x
        x = torch.cat([route, x], dim = 1)
        x = self.maxpool(x)
        return x,feat
class CSPDarkNet(nn.Module):
    def __init__(self):
        super(CSPDarkNet, self).__init__()
        self.conv1 = conv_dw(3, 32, stride=2)
        self.conv2 = conv_dw(32, 64, stride=2)
        self.resblock_body1 =  Resblock_body(64, 64)
        self.resblock_body2 =  Resblock_body(128, 128)
        self.resblock_body3 =  Resblock_body(256, 256)
        self.conv3 = conv_dw(512, 512)
        self.num_features = 1
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        feat4 = x
        x, _    = self.resblock_body1(x)
        x, feat3    = self.resblock_body2(x)
        x, feat1    = self.resblock_body3(x)
        x = self.conv3(x)
        feat2 = x
        return feat1,feat2,feat3,feat4
def darknet53_tiny(pretrained, **kwargs):
    model = CSPDarkNet()
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model
