# -*- coding: utf-8 -*-
# @Time    : 2018/12/26 16:57
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : params.py
# @Software: PyCharm


import os
import torch
import logging  # 引入logging模块
logging.basicConfig(level=logging.NOTSET)  # 设置日志级别

class Args:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 检测是否可以使用gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 32
    num_epochs = 50
    test_freq = 1
    save_freq = 1
    weight_reg = 0.01
    K = 10