import torch
import torch.nn as nn
import torch.nn.functional as F
from smt import smt_t

from thop import profile
from torch import Tensor
from typing import List


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
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

class HMU(nn.Module):
    def __init__(self, in_c, num_groups=4, hidden_dim=None, in_ch=-1):
        super().__init__()
        self.num_groups = num_groups

        hidden_dim = hidden_dim or in_c // 2
        expand_dim = hidden_dim * num_groups
        self.expand_conv = BasicConv2d(in_c, expand_dim, 1)
        self.gate_genator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(num_groups * hidden_dim, hidden_dim, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, num_groups * hidden_dim, 1),
            nn.Softmax(dim=1),
        )

        self.interact = nn.ModuleDict()
        self.interact["0"] = BasicConv2d(hidden_dim, 2 * hidden_dim, 3, 1, 1)
        for group_id in range(1, num_groups - 1):
            self.interact[str(group_id)] = BasicConv2d(2 * hidden_dim, 3 * hidden_dim, 3, 1, 1)
        self.conv1 = BasicConv2d(4 * hidden_dim, 2 * hidden_dim, 3, 1, 1)
        self.conv2 = BasicConv2d(4 * hidden_dim, 4 * hidden_dim, 3, 1, 1)
        self.interact["2"] = BasicConv2d(4 * hidden_dim, 2 * hidden_dim, 3, 1, 1)
        self.interact["1"] = BasicConv2d(hidden_dim, 2 * hidden_dim, 3, 1, 1)
        self.interact["3"] = BasicConv2d(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)

        self.fuse = nn.Sequential(nn.Conv2d(num_groups * hidden_dim, in_c, 3, 1, 1), nn.BatchNorm2d(in_c))
        self.final_relu = nn.ReLU(True)

    def forward(self, x):
        xs = self.expand_conv(x).chunk(self.num_groups, dim=1)

        outs = []
        # x1 = xs[0]
        branch_out0 = self.interact["0"](xs[0])

        # outs.append(branch_out.chunk(3, dim=1))
        # x01, x02, x03 = branch_out0.chunk(3, dim=1)

        branch_out1 = self.interact["1"](xs[1])
        branch_out2 = self.interact["1"](xs[2])
        branch_out3 = self.interact["1"](xs[3])
        branch_out01 = torch.cat((branch_out0, branch_out1), 1)
        # print(branch_out01.shape[1])
        branch_out01 = self.interact["2"](branch_out01)
        branch_out012 = torch.cat((branch_out01, branch_out2), 1)
        branch_out012 = self.interact["2"](branch_out012)
        branch_out0123 = torch.cat((branch_out012, branch_out3), 1)
        branch_out0123 = self.interact["2"](branch_out0123)

        out21 = torch.cat((branch_out01, branch_out0), 1)
        out21 = self.conv1(out21)
        out22 = torch.cat((branch_out01, branch_out012), 1)
        out22 = self.conv1(out22)
        out23 = torch.cat((branch_out0123, branch_out012), 1)
        out23 = self.conv1(out23)
        branch_out22 = torch.cat((out21, out22), 1)
        branch_out22 = self.conv1(branch_out22)
        branch_out23 = torch.cat((branch_out22, out23), 1)
        branch_out23 = self.conv1(branch_out23)

        out31 = torch.cat((branch_out22, out21), 1)
        out31 = self.conv1(out31)
        out32 = torch.cat((branch_out23, branch_out22), 1)
        out32 = self.conv1(out32)
        branch_out32 = torch.cat((out31, out32), 1)
        branch_out32 = self.conv1(branch_out32)

        out = torch.cat((out31, branch_out32), 1)
        out = self.conv2(out)

        gate = self.gate_genator(out)
        out = self.fuse(out * gate)
        return self.final_relu(out + x)


class MBR(nn.Module):
    expansion = 1  # 类属性，目前设定为1，但没有在代码中直接使用。这个属性在某些网络结构中用于标识特征图扩展的倍数。

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(MBR, self).__init__()
        # branch1
        self.atrConv = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, dilation=3, padding=3, stride=1), nn.BatchNorm2d(planes), nn.PReLU()
        )
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1,  # 使用3x3的卷积核，保持输入输出尺寸不变
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)  # 批量归一化，稳定和加速训练
        self.relu = nn.ReLU(inplace=True)  # 在地方使用，增加非线性
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample1 = upsample
        self.stride1 = stride
        # barch2
        self.conv3 = nn.Conv2d(inplanes, inplanes, kernel_size=5, stride=1,
                               padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.relu3 = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv4 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv4 = nn.Conv2d(inplanes, planes, kernel_size=5, stride=stride,
                                   padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.conv_cat = BasicConv2d(3 * inplanes, inplanes, 3, padding=1)
        self.upsample2 = upsample
        self.stride2 = stride


    def forward(self, x):
        residual = x

        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out1 = self.conv2(out1)
        out1 = self.bn2(out1)

        if self.upsample1 is not None:
            residual = self.upsample1(x)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out2 = self.relu3(out2)

        out2 = self.conv4(out2)
        out2 = self.bn4(out2)
        out3 = self.atrConv(x)
        if self.upsample2 is not None:
            residual = self.upsample2(x)
        out = self.conv_cat(torch.cat((out1, out2, out3), 1))
        out += residual
        out = self.relu(out)

        return out

class PRNet(nn.Module):
    def __init__(self, channel=64):
        super(PRNet, self).__init__()

        self.smt = smt_t()
        self.up2 = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor = 4)

        # self.UE1 = GCM(512, 512)
        # self.UE2 = GCM(256, 256)
        # self.UE3 = GCM(128, 128)
        # self.UE4 = GCM(64, 64)

        self.Translayer2_1 = BasicConv2d(128, 64, 1)
        self.Translayer3_1 = BasicConv2d(256, 64, 1)
        self.Translayer4_1 = BasicConv2d(512, 64, 1)

        self.MBR = MBR(64, 64)

        self.d4 = nn.Sequential(HMU(64, num_groups=4, hidden_dim=32))
        self.d3 = nn.Sequential(HMU(64, num_groups=4, hidden_dim=32))
        self.d2 = nn.Sequential(HMU(64, num_groups=4, hidden_dim=32))
        self.d1 = nn.Sequential(HMU(64, num_groups=4, hidden_dim=32))

        self.uconv1 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1, relu=True)
        self.uconv2 = BasicConv2d(128, 64, 1)
        self.uconv3 = BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1, relu=True)

        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.predtrans2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)


    def forward(self,x):
        #print('01', datetime.now())
        rgb_list = self.smt(x)

        r1 = rgb_list[3]  # 512,12
        r2 = rgb_list[2]  # 256,24
        r3 = rgb_list[1]  # 128,48
        r4 = rgb_list[0]  # 64,96,96

        r3 = self.Translayer2_1(r3)  # [1, 64, 44, 44]
        r2 = self.Translayer3_1(r2)
        r1 = self.Translayer4_1(r1)  # 都变为64的通道

        r1 = self.up1(r1)  # 将图像大小扩大一倍
        x1 = self.d4(r2 + r1)
        x1 = self.MBR(x1)
        x1 = self.up1(x1)  # 将图像大小扩大一倍
        x12 = self.d3(x1 + r3)
        x12 = self.MBR(x12)
        x12 = self.up1(x12)  # 将图像大小扩大一倍
        # x1 = self.up1(x1)  # 将图像大小扩大一倍
        x123 = self.d2(x12 + r4)
        x123 = self.MBR(x123)
        # x123 = self.up1(x123)  # 将图像大小扩大一倍
        # r4 = self.up1(r4)  # 将图像大小扩大一倍
        # x1234 = self.d1(x123 + r4)
        # x1234 = self.MBR(x1234)

        # r1234 = F.interpolate(self.predtrans2(x1234), size=416, mode='bilinear')
        r123 = F.interpolate(self.predtrans2(x123), size=416, mode='bilinear')
        r12 = F.interpolate(self.predtrans3(x12), size=416, mode='bilinear')
        r1 = F.interpolate(self.predtrans4(x1), size=416, mode='bilinear')

        return r123, r12, r1

    def load_pre(self, pre_model):
        self.smt.load_state_dict(torch.load(pre_model)['model'])
        print(f"loading pre_model ${pre_model}")



if __name__ == '__main__':
    x = torch.randn(1, 3, 416, 416)
    flops, params = profile(PRNet(x), (x,))
    print('flops: %.2f G, parms: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
