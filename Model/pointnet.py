# -*- coding: utf-8 -*-
# @Time    : 2018/11/8 8:26
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : pointnet.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from torchsummary import summary

from Model.TransformNet import InputTransformNet, FeatureTransformNet

def conv_bn_block(input, output, kernel_size):
    '''
    标准卷积块（conv + bn + relu）
    :param input: 输入通道数
    :param output: 输出通道数
    :param kernel_size: 卷积核大小
    :return:
    '''
    return nn.Sequential(
        nn.Conv1d(input, output, kernel_size),
        nn.BatchNorm1d(output),
        nn.ReLU(inplace=True)
    )


def fc_bn_block(input, output):
    '''
    标准全连接块（fc + bn + relu）
    :param input:  输入通道数
    :param output:  输出通道数
    :return:  卷积核大小
    '''
    return nn.Sequential(
        nn.Linear(input, output),
        nn.BatchNorm1d(output),
        nn.ReLU(inplace=True)
    )

class PointNet(nn.Module):
    def __init__(self, num_classes=40):
        super(PointNet, self).__init__()

        self.num_classes = num_classes

        self.t_net_1 = InputTransformNet()
        self.conv_block_1 = conv_bn_block(3, 64, 1)
        self.conv_block_2 = conv_bn_block(64, 64, 1)

        self.t_net_3 = FeatureTransformNet()
        self.conv_block_4 = conv_bn_block(64, 64, 1)
        self.conv_block_5 = conv_bn_block(64, 128, 1)
        self.conv_block_6 = conv_bn_block(128, 1024, 1)

        self.fc_block_7 = fc_bn_block(1024, 512)
        self.drop_7 = nn.Dropout(0.7)
        self.fc_block_8 = fc_bn_block(512, 256)
        self.drop_8 = nn.Dropout(0.7)
        self.fc_9 = nn.Linear(256, self.num_classes)

        # 使用xavier初始化参数
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, N = int(x.shape[0]), int(x.shape[2])

        input_transform = self.t_net_1(x)
        x = torch.matmul(x.permute(0, 2, 1), input_transform.permute(0, 2, 1)).permute(0, 2, 1)

        x = self.conv_block_1(x)
        x = self.conv_block_2(x)

        feature_transform = self.t_net_3(x)
        x = torch.matmul(x.permute(0, 2, 1), feature_transform.permute(0, 2, 1)).permute(0, 2, 1)

        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.conv_block_6(x)

        x = nn.MaxPool1d(N)(x)
        x = x.view(B, 1024)

        x = self.drop_7(self.fc_block_7(x))
        x = self.drop_8(self.fc_block_8(x))
        x = self.fc_9(x)
        x = F.log_softmax(x, dim=-1)

        return x


class PointNet_Vanilla(nn.Module):
    def __init__(self, num_classes=40):
        super(PointNet_Vanilla, self).__init__()

        self.num_classes = num_classes

        self.conv_block_1 = conv_bn_block(3, 64, 1)
        self.conv_block_2 = conv_bn_block(64, 64, 1)

        self.conv_block_4 = conv_bn_block(64, 64, 1)
        self.conv_block_5 = conv_bn_block(64, 128, 1)
        self.conv_block_6 = conv_bn_block(128, 1024, 1)

        self.fc_block_7 = fc_bn_block(1024, 512)
        self.drop_7 = nn.Dropout(0.5)
        self.fc_block_8 = fc_bn_block(512, 256)
        self.drop_8 = nn.Dropout(0.5)
        self.fc_9 = nn.Linear(256, self.num_classes)

        # 使用xavier初始化参数
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        B, N = int(x.shape[0]), int(x.shape[2])

        x = self.conv_block_1(x)
        x = self.conv_block_2(x)

        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.conv_block_6(x)

        x = nn.MaxPool1d(N)(x)
        x = x.view(B, 1024)

        x = self.drop_7(self.fc_block_7(x))
        x = self.drop_8(self.fc_block_8(x))
        x = self.fc_9(x)
        x = F.log_softmax(x, dim=-1)

        return x


if __name__ == '__main__':
    # net = PointNet(num_classes=40)
    # net = PointNet_Vanilla(num_classes=40)
    # summary(net, (3, 2048))

    dummy_input = torch.randn([5, 50, 3])
    print('input shape: {}'.format(dummy_input.shape))
    print('input: {}'.format(dummy_input))
    model = PointNet_Vanilla(num_classes=40)
    out = model(dummy_input)
    print('out shape: {}'.format(out.shape))
    print('out: {}'.format(out))




