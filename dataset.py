# -*- coding: utf-8 -*-
# @Time    : 2018/11/8 7:40
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : dataset.py
# @Software: PyCharm

import os
import torch.utils.data as data
import numpy as np
import h5py
import logging  # 引入logging模块
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

modelnet_dir = 'F:\DataSets\modelnet40_ply_hdf5_2048'

class ModelNet40DataSet(data.Dataset):
    def __init__(self, train=True):
        if train:
            data_file = os.path.join(modelnet_dir, 'train_files.txt')
        else:
            data_file = os.path.join(modelnet_dir, 'test_files.txt')

        with open(data_file, 'r') as f:
            file_list = [line.strip().split('/')[-1] for line in f.readlines()]

        logging.info(file_list)

        modelnet_data = np.zeros([0, 2048, 3], dtype=np.float32)
        modelnet_label = np.zeros([0, 1], np.int64)

        for file_name in file_list:
            file_path = os.path.join(modelnet_dir, file_name)
            file = h5py.File(file_path)
            # print(file['data'].shape)
            # print(file['label'].shape)
            data = file['data'][:]
            label = file['label'][:]

            modelnet_data = np.concatenate([modelnet_data, data], axis=0)
            modelnet_label = np.concatenate([modelnet_label, label], axis=0)

        logging.info('modelnet_data: {}'.format(modelnet_data.shape))
        logging.info('modelnet_label: {}'.format(modelnet_label.shape))

        self.point_cloud = modelnet_data
        self.label = modelnet_label

    def __getitem__(self, item):
        return self.point_cloud[item], self.label[item]

    def __len__(self):
        return self.label.shape[0]




if __name__ == '__main__':
    modelnet_train = ModelNet40DataSet(train=True)
    modelnet_test = ModelNet40DataSet(train=False)

    data, label = modelnet_train[0]
    print('type: {}'.format(type(data)))
    print('type: {}'.format(type(label)))
    print(label)