# -*- coding: utf-8 -*-
# @Time    : 2018/11/8 9:03
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : train.py
# @Software: PyCharm

import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
import logging  # 引入logging模块
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
import warnings
warnings.filterwarnings("ignore")

from dataset import ModelNet40DataSet
from Model.dynami_graph_cnn import DGCNNCls_vanilla, DGCNNCls_vanilla_2
from Model.pointnet import PointNet_Vanilla
from Model.l2_reg_loss import L2RegularizationLoss
from params import Args


class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train():
    pwd = os.getcwd()
    weights_dir = os.path.join(pwd, 'weights')
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    logging.info('Loading Dataset...')
    train_dataset = ModelNet40DataSet(train=True)
    test_dataset = ModelNet40DataSet(train=False)
    logging.info('train_dataset: {}'.format(len(train_dataset)))
    logging.info('test_dataset: {}'.format(len(test_dataset)))
    logging.info('Done...\n')


    logging.info('Creating DataLoader...')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=Args.batch_size, shuffle=False, num_workers=2)
    logging.info('Done...\n')


    logging.info('Checking gpu...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        logging.info('gpu available: {}'.format(torch.cuda.device_count()))
        logging.info('current gpu: {}'.format(torch.cuda.get_device_name(0)))
        logging.info('gpu capability: {}'.format(torch.cuda.get_device_capability(0)))
    else:
        logging.info('gpu not available, running on cpu instead.')
    logging.info('Done...\n')


    logging.info('Create SummaryWriter in ./summary')
    # 创建SummaryWriter
    summary_writer = SummaryWriter(comment='PointNet', log_dir='summary')
    logging.info('Done...\n')


    logging.info('Creating Model...')
    # create DGCNN
    model = DGCNNCls_vanilla(num_classes=40).to(Args.device)
    # model = DGCNNCls_vanilla_2(num_classes=40).to(Args.device)
    # create pointnet
    # model = PointNet_Vanilla(num_classes=40).to(Args.device)

    # add graph
    # dummy_input = torch.rand(2, 2048, 3).to(device)
    # summary_writer.add_graph(model, dummy_input)

    # CrossEntropy Loss
    criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss()
    # 正则化损失
    # criterion_reg_loss = L2RegularizationLoss().to(Args.device)
    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=Args.weight_reg)
    # 学习率衰减
    schedular = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    logging.info('Done...\n')


    logging.info('Start training...')
    for epoch in range(1, Args.num_epochs+1):
        logging.info("--------Epoch {}--------".format(epoch))

        # 更新学习率
        schedular.step()

        tqdm_batch = tqdm(train_loader, desc='Epoch-{} training'.format(epoch))

        # train
        model.train()
        # crossentropy_loss_tracker = AverageMeter()
        # reg_loss_tracker = AverageMeter()
        loss_tracker = AverageMeter()
        for batch_idx, (data, label) in enumerate(tqdm_batch):
            data, label = data.to(device), label.to(device)
            # print(data.size())
            # data = data.permute(0, 2, 1)

            out = model(data)

            # print('out: {}, label: {}'.format(out.size(), label.size()))
            loss = criterion(out, label.view(-1))
            # reg_loss = Args.weight_reg * criterion_reg_loss(model.parameters())
            # loss = crossentropy_loss + reg_loss
            # loss = crossentropy_loss

            optimizer.zero_grad()
            # loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()

            # crossentropy_loss_tracker.update(crossentropy_loss.item(), label.size(0))
            # reg_loss_tracker.update(reg_loss, label.size(0))
            loss_tracker.update(loss.item(), label.size(0))

            # print('\n')
            # print('cross_entropy_loss: {}'.format(crossentropy_loss.item()))
            # print('reg_loss: {}'.format(reg_loss))
            # print('loss: {}'.format(loss.item()))
            # print('\n')

            # del data, label

        tqdm_batch.close()
        # logging.info('Crossentropy Loss: {:.4f} ({:.4f})'.format(crossentropy_loss_tracker.val, crossentropy_loss_tracker.avg))
        # logging.info('Reg Loss: {:.4f} ({:.4f})'.format(reg_loss_tracker.val, reg_loss_tracker.avg))
        logging.info('Loss: {:.4f} ({:.4f})'.format(loss_tracker.val, loss_tracker.avg))

        summary_writer.add_scalar('loss', loss_tracker.avg, epoch)

        if epoch % Args.test_freq == 0:
            tqdm_batch = tqdm(test_loader, desc='Epoch-{} testing'.format(epoch))

            model.eval()
            correct_cnt = 0
            total_cnt = 0
            with torch.no_grad():
                for batch_idx, (data, label) in enumerate(tqdm_batch):
                    data, label = data.to(device), label.to(device)
                    # data = data.permute(0, 2, 1)

                    out = model(data)
                    pred_choice = out.max(1)[1]

                    correct_cnt += pred_choice.eq(label.view(-1)).sum().item()
                    total_cnt += label.size(0)

                    # del data, label

            print('correct_cnt: {}, total_cnt: {}'.format(correct_cnt, total_cnt))
            acc = correct_cnt / total_cnt
            logging.info('Accuracy: {:.4f}'.format(acc))

            summary_writer.add_scalar('acc', acc, epoch)

            tqdm_batch.close()

        if epoch % Args.save_freq == 0:
            ckpt_name = os.path.join(weights_dir, 'dgcnn_{0}.pth'.format(epoch))
            torch.save(model.state_dict(), ckpt_name)
            logging.info('model saved in {}'.format(ckpt_name))

    summary_writer.close()


if __name__ == '__main__':
    train()