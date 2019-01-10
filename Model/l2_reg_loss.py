# -*- coding: utf-8 -*-
# @Time    : 2018/12/19 2:25
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : l2_reg_loss.py
# @Software: PyCharm

import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary
from params import Args


class L2RegularizationLoss(nn.Module):
    def __init__(self):
        super(L2RegularizationLoss, self).__init__()

        self.l2_reg = torch.tensor(0.).to(Args.device)

    def forward(self, params):
        self.l2_reg = torch.tensor(0.).to(Args.device)
        for param in params:
            # print(param.dtype)
            self.l2_reg += torch.norm(param)
        return self.l2_reg

