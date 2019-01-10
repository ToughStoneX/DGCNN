# -*- coding: utf-8 -*-
# @Time    : 2018/12/26 15:59
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : DGCNN.py
# @Software: PyCharm

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from torchsummary import summary
from collections import OrderedDict


from params import Args


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

class EdgeConv(nn.Module):
    '''
    EdgeConv模块
    1. 输入为：n * f
    2. 创建KNN graph，变为： n * k * f
    3. 接上若干个mlp层：a1, a2, ..., an
    4. 最终输出为：n * k * an
    5. 全局池化，变为： n * an
    '''
    def __init__(self, layers, K=20):
        '''
        构造函数
        :param layers: e.p. [3, 64, 64, 64]
        :param K:
        '''
        super(EdgeConv, self).__init__()

        self.K = K
        self.layers = layers
        # self.KNN_Graph = torch.zeros(Args.batch_size, 2048, self.K, self.layers[0]).to(Args.device)

        if layers is None:
            self.mlp = None
        else:
            mlp_layers = OrderedDict()
            for i in range(len(self.layers) - 1):
                if i == 0:
                    mlp_layers['conv_bn_block_{}'.format(i + 1)] = conv_bn_block(2*self.layers[i], self.layers[i + 1], 1)
                else:
                    mlp_layers['conv_bn_block_{}'.format(i+1)] = conv_bn_block(self.layers[i], self.layers[i+1], 1)
            self.mlp = nn.Sequential(mlp_layers)

    def createSingleKNNGraph(self, X):
        '''
        generate a KNN graph for a single point cloud
        :param X:  X is a Tensor, shape: [N, F]
        :return: KNN graph, shape: [N, K, F]
        '''
        N, F = X.shape
        assert F == self.layers[0]

        # self.KNN_Graph = np.zeros(N, self.K)

        # 计算距离矩阵
        dist_mat = torch.pow(X, 2).sum(dim=1, keepdim=True).expand(N, N) + \
                   torch.pow(X, 2).sum(dim=1, keepdim=True).expand(N, N).t()
        dist_mat.addmm_(1, -2, X, X.t())

        # 对距离矩阵排序
        dist_mat_sorted, sorted_indices = torch.sort(dist_mat, dim=1)
        # print(dist_mat_sorted)

        # 取出前K个（除去本身）
        knn_indexes = sorted_indices[:, 1:self.K+1]
        # print(sorted_indices)

        # 创建KNN图
        knn_graph = X[knn_indexes]

        return knn_graph

    def forward(self, X):
        '''
        前向传播函数
        :param X:  shape: [B, N, F]
        :return:  shape: [B, N, an]
        '''
        # print(X.shape)
        B, N, F = X.shape
        assert F == self.layers[0]

        KNN_Graph = torch.zeros(B, N, self.K, self.layers[0]).to(Args.device)

        # creating knn graph
        # X: [B, N, F]
        for idx, x in enumerate(X):
            # x: [N, F]
            # knn_graph: [N, K, F]
            # self.KNN_Graph[idx] = self.createSingleKNNGraph(x)
            KNN_Graph[idx] = self.createSingleKNNGraph(x)
        # print(self.KNN_Graph.shape)
        # print('KNN_Graph: {}'.format(KNN_Graph[0][0]))

        # X: [B, N, F]
        x1 = X.reshape([B, N, 1, F])
        x1 = x1.expand(B, N, self.K, F)
        # x1: [B, N, K, F]

        x2 = KNN_Graph - x1
        # x2: [B, N, K, F]

        x_in = torch.cat([x1, x2], dim=3)
        # x_in: [B, N, K, 2*F]
        x_in = x_in.permute(0, 3, 1, 2)
        # x_in: [B, 2*F, N, K]

        # reshape, x_in: [B, 2*F, N*K]
        x_in = x_in.reshape([B, 2 * F, N * self.K])

        # out: [B, an, N*K]
        out = self.mlp(x_in)
        _, an, _ = out.shape
        # print(out.shape)

        out = out.reshape([B, an, N, self.K])
        # print(out.shape)
        # reshape, out: [B, an, N, K]
        out = out.reshape([B, an*N, self.K])
        # print(out.shape)
        # reshape, out: [B, an*N, K]
        out = nn.MaxPool1d(self.K)(out)
        # print(out.shape)
        out = out.reshape([B, an, N])
        # print(out.shape)
        out = out.permute(0, 2, 1)
        # print(out.shape)

        return out


class DGCNNCls_vanilla(nn.Module):
    def __init__(self, num_classes):
        super(DGCNNCls_vanilla, self).__init__()

        self.num_classes = num_classes
        self.edge_conv_1 = EdgeConv(layers=[3, 64, 64, 64], K=Args.K)
        self.edge_conv_2 = EdgeConv(layers=[64, 128], K=Args.K)
        self.conv_block_3 = conv_bn_block(128, 1024, 1)
        self.fc_block_4 = fc_bn_block(1024, 512)
        self.drop_4 = nn.Dropout(0.5)
        self.fc_block_5 = fc_bn_block(512, 256)
        self.drop_5 = nn.Dropout(0.5)
        self.fc_6 = nn.Linear(256, self.num_classes)

        # 初始化参数
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
        '''
        前向传播
        :param x: shape: [B, N, 3]
        :return:
        '''
        # print(x.shape)
        B, N, C = x.shape
        assert C == 3, 'dimension of x does not match'
        # x: [B, N, 3]
        x = self.edge_conv_1(x)
        # x: [B, N, 64]
        x = self.edge_conv_2(x)
        # x: [B, N, 128]
        x = x.permute(0, 2, 1)
        # x: [B, 128, N]
        x = self.conv_block_3(x)
        # x: [B, 1024, N]
        # x = x.permute(0, 2, 1)
        # x: [B, N, 1024]
        x = nn.MaxPool1d(N)(x)
        # print(x.shape)
        # x: [B, 1, 1024]
        x = x.reshape([B, 1024])
        # x: [B, 1024]
        x = self.fc_block_4(x)
        x = self.drop_4(x)
        # x: [B, 512]
        x = self.fc_block_5(x)
        x = self.drop_5(x)
        # x: [B, 256]
        x = self.fc_6(x)

        # softmax
        x = F.log_softmax(x, dim=-1)

        return x


class DGCNNCls_vanilla_2(nn.Module):
    def __init__(self, num_classes):
        super(DGCNNCls_vanilla_2, self).__init__()

        self.num_classes = num_classes
        self.edge_conv_1 = EdgeConv(layers=[3, 32, 64, 64], K=Args.K)
        self.edge_conv_2 = EdgeConv(layers=[64, 128, 256], K=Args.K)
        self.conv_block_3 = conv_bn_block(256, 1024, 1)
        self.fc_block_4 = fc_bn_block(1024, 512)
        self.drop_4 = nn.Dropout(0.5)
        self.fc_block_5 = fc_bn_block(512, 256)
        self.drop_5 = nn.Dropout(0.5)
        self.fc_6 = nn.Linear(256, self.num_classes)

        # 初始化参数
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
        '''
        前向传播
        :param x: shape: [B, N, 3]
        :return:
        '''
        # print(x.shape)
        B, N, C = x.shape
        assert C == 3, 'dimension of x does not match'
        # x: [B, N, 3]
        x = self.edge_conv_1(x)
        # x: [B, N, 64]
        x = self.edge_conv_2(x)
        # x: [B, N, 128]
        x = x.permute(0, 2, 1)
        # x: [B, 128, N]
        x = self.conv_block_3(x)
        # x: [B, 1024, N]
        # x = x.permute(0, 2, 1)
        # x: [B, N, 1024]
        x = nn.MaxPool1d(N)(x)
        # print(x.shape)
        # x: [B, 1, 1024]
        x = x.reshape([B, 1024])
        # x: [B, 1024]
        x = self.fc_block_4(x)
        x = self.drop_4(x)
        # x: [B, 512]
        x = self.fc_block_5(x)
        x = self.drop_5(x)
        # x: [B, 256]
        x = self.fc_6(x)

        # softmax
        x = F.log_softmax(x, dim=-1)

        return x



if __name__ == '__main__':
    # dummy_input = torch.randn([5, 50, 3])
    # print('input shape: {}'.format(dummy_input.shape))
    # print('input: {}'.format(dummy_input))
    # model = EdgeConv(layers=[3, 16, 32], K=20).to(Args.device)
    # out = model(dummy_input)
    # print('out shape: {}'.format(out.shape))
    # print('out: {}'.format(out))

    # net = DGCNNCls_vanilla(num_classes=40)
    # summary(net, (100, 3))


    dummy_input = torch.randn([5, 50, 3])
    print('input shape: {}'.format(dummy_input.shape))
    # print('input: {}'.format(dummy_input))
    # model = DGCNNCls_vanilla(num_classes=40)
    model = DGCNNCls_vanilla_2(num_classes=40)
    out = model(dummy_input)
    print('out shape: {}'.format(out.shape))
    # print('out: {}'.format(out))










