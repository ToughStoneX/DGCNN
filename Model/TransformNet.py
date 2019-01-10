# -*- coding: utf-8 -*-
# @Time    : 2018/11/8 8:07
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : TransformNet.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


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


class InputTransformNet(nn.Module):
    def __init__(self):
        super(InputTransformNet, self).__init__()

        self.conv_block_1 = conv_bn_block(3, 64, 1)
        self.conv_block_2 = conv_bn_block(64, 128, 1)
        self.conv_block_3 = conv_bn_block(128, 1024, 1)

        self.fc_block_4 = fc_bn_block(1024, 512)
        self.fc_block_5 = fc_bn_block(512, 256)

        self.transform = nn.Linear(256, 3*3)
        #  转换矩阵初始化
        # transform层的weight为[256 * 9]，bias为9；weight全部初始化为0， bias初始为[1, 0, 0, 0, 1, 0, 0, 0, 1]
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        B, N = int(x.shape[0]), int(x.shape[2])

        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)

        x = nn.MaxPool1d(N)(x)

        x = x.view(B, 1024)
        x = self.fc_block_4(x)
        x = self.fc_block_5(x)

        x = self.transform(x)
        x = x.view(B, 3, 3)

        return x

class FeatureTransformNet(nn.Module):
    def __init__(self):
        super(FeatureTransformNet, self).__init__()

        self.conv_block_1 = conv_bn_block(64, 64, 1)
        self.conv_block_2 = conv_bn_block(64, 128, 1)
        self.conv_block_3 = conv_bn_block(128, 1024, 1)

        self.fc_block_4 = fc_bn_block(1024, 512)
        self.fc_block_5 = fc_bn_block(512, 256)

        self.transform = nn.Linear(256, 64*64)
        #  转换矩阵初始化
        # transform层的weight为[256 * 9]，bias为9；weight全部初始化为0， bias初始为[1, 0, 0, 0, 1, 0, 0, 0, 1]
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(64, 64))

    def forward(self, x):
        B, N = int(x.shape[0]), int(x.shape[2])

        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)

        x = nn.MaxPool1d(N)(x)

        x = x.view(B, 1024)
        x = self.fc_block_4(x)
        x = self.fc_block_5(x)

        x = self.transform(x)
        x = x.view(B, 64, 64)

        return x


if __name__ == '__main__':
    a = torch.rand(8, 3, 1000)
    t = InputTransformNet()
    x = t(a)
    print(x.shape)

    b = torch.rand(8, 64, 1000)
    t2 = FeatureTransformNet()
    x = t2(b)
    print(x.shape)


