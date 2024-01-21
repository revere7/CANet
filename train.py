#!/usr/bin/env python3

from distutils.errors import DistutilsModuleError
import os
import argparse
import numpy as np
# import pandas as pd
import cv2
from pathlib import Path
import copy
import logging
import random
import scipy.io as sio
# import matplotlib.pyplot as plt
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from srm_filter_kernel import all_normalized_hpf_list

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from MPNCOV.python import MPNCOV
from CANet import CANet

IMAGE_SIZE = 256
BATCH_SIZE = 27
LR = 0.02
WEIGHT_DECAY = 5e-4

EVAL_PRINT_FREQUENCY = 1
OUTPUT_PATH = Path(__file__).stem

class AverageMeter(object):
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


def train(model, device, train_loader, optimizer, epoch):
    batch_time = AverageMeter()  # ONE EPOCH TRAIN TIME
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()

    for i, sample in enumerate(train_loader):

        data_time.update(time.time() - end)

        data, label = sample['data'], sample['label']

        shape = list(data.size())
        data = data.reshape(shape[0] * shape[1], *shape[2:])
        label = label.reshape(-1)

        data, label = data.to(device), label.to(device)
        # data, label = data.cuda(), label.cuda()

        optimizer.zero_grad()

        end = time.time()

        output = model(data)  # FP

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, label)

        losses.update(loss.item(), data.size(0))

        loss.backward()  # BP
        optimizer.step()

        batch_time.update(time.time() - end)  # BATCH TIME = BATCH BP+FP
        end = time.time()

        if i % TRAIN_PRINT_FREQUENCY == 0:
            # logging.info('Epoch: [{}][{}/{}] \t Loss {:.6f}'.format(epoch, i, len(train_loader), loss.item()))

            logging.info('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))


def adjust_bn_stats(model, device, train_loader):
    model.train()

    with torch.no_grad():
        for sample in train_loader:
            data, label = sample['data'], sample['label']

            shape = list(data.size())
            data = data.reshape(shape[0] * shape[1], *shape[2:])
            label = label.reshape(-1)

            data, label = data.to(device), label.to(device)
            # data, label = data.cuda(), label.cuda()

            output = model(data)


def evaluate(model, device, eval_loader, epoch, optimizer, best_acc, PARAMS_PATH, PARAMS_PATH1, TMP):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for sample in eval_loader:
            data, label = sample['data'], sample['label']

            shape = list(data.size())
            data = data.reshape(shape[0] * shape[1], *shape[2:])
            label = label.reshape(-1)

            data, label = data.to(device), label.to(device)
            # data, label = data.cuda(), label.cuda()

            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()

    accuracy = correct / (len(eval_loader.dataset) * 2)

    all_state = {
        'original_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch
      }
    torch.save(all_state, PARAMS_PATH1)

    if accuracy > best_acc and epoch > TMP:
        best_acc = accuracy
        all_state = {
            'original_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(all_state, PARAMS_PATH)

    logging.info('-' * 8)
    logging.info('Eval accuracy: {:.4f}'.format(accuracy))
    logging.info('Best accuracy:{:.4f}'.format(best_acc))
    logging.info('-' * 8)
    return best_acc


def initWeights(module):
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

    if type(module) == nn.Linear:
        nn.init.normal_(module.weight.data, mean=0, std=0.01)
        #nn.init.constant_(module.bias.data, val=0)


class AugData():
    def __call__(self, sample):
        data, label = sample['data'], sample['label']

        rot = random.randint(0, 3)

        data = np.rot90(data, rot, axes=[2, 3]).copy()  # gaiwei [2,3]

        if random.random() < 0.5:
            data = np.flip(data, axis=2).copy()

        new_sample = {'data': data, 'label': label}

        return new_sample


class ToTensor():
    def __call__(self, sample):
        data, label = sample['data'], sample['label']

        # data = np.expand_dims(data, axis=1) ##delete
        data = data.astype(np.float32)
        # data = data / 255.0

        new_sample = {
            'data': torch.from_numpy(data),
            'label': torch.from_numpy(label).long(),
        }

        return new_sample


class MyDataset(Dataset):
    def __init__(self, index_path, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, transform=None):
        self.index_list = np.load(index_path)
        self.transform = transform

        self.bossbase_cover_path = BOSSBASE_COVER_DIR + '/{}.ppm'
        self.bossbase_stego_path = BOSSBASE_STEGO_DIR + '/{}.ppm'

    def __len__(self):
        return self.index_list.shape[0]

    def __getitem__(self, idx):
        file_index = self.index_list[idx]

        cover_path = self.bossbase_cover_path.format(file_index)
        stego_path = self.bossbase_stego_path.format(file_index)

        cover_data = cv2.imread(cover_path, -1)
        cover_data = cv2.cvtColor(cover_data, cv2.COLOR_BGR2RGB)
        cover_data = np.transpose(cover_data, (2, 0, 1))
        stego_data = cv2.imread(stego_path, -1)
        stego_data = cv2.cvtColor(stego_data, cv2.COLOR_BGR2RGB)
        stego_data = np.transpose(stego_data, (2, 0, 1))

        data = np.stack([cover_data, stego_data])
        label = np.array([0, 1], dtype='int32')

        sample = {'data': data, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


def setLogger(log_path, mode='a'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, mode=mode)
        file_handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def main(args):
    statePath = args.statePath
    device = torch.device("cuda")
    kwargs = {'num_workers': 0, 'pin_memory': True}

    train_transform = transforms.Compose([
        AugData(),
        ToTensor()
    ])

    eval_transform = transforms.Compose([
        ToTensor()
    ])

    DATASET_INDEX = args.DATASET_INDEX
    STEGANOGRAPHY = args.STEGANOGRAPHY
    EMBEDDING_RATE = args.EMBEDDING_RATE
    TIMES = args.times


    BOSSBASE_COVER_DIR = '/data/ALASKA_v2_TIFF_256_COLOR'
    BOSSBASE_STEGO_DIR = '/data/ALASKA_v2_TIFF_256_COLOR_HILL-GINA_0.4'

    TRAIN_INDEX_PATH = 'index_list/alaska_train_index_14000.npy'
    VALID_INDEX_PATH = 'index_list/alaska_val_index_1000.npy'
    TEST_INDEX_PATH = 'index_list/alaska_test_index_5000.npy'

    LOAD_RATE = float(EMBEDDING_RATE) + 0.1
    LOAD_RATE = round(LOAD_RATE, 1)

    PARAMS_NAME = '{}-{}-{}-params-{}-lr={}.pt'.format(STEGANOGRAPHY, EMBEDDING_RATE, DATASET_INDEX, TIMES,LR)
    PARAMS_NAME1 = '{}-{}-{}-process-params-{}-lr={}.pt'.format(STEGANOGRAPHY, EMBEDDING_RATE, DATASET_INDEX, TIMES,LR)
    LOG_NAME = '{}-{}-{}-model_log-{}-lr={}.log'.format(STEGANOGRAPHY, EMBEDDING_RATE, DATASET_INDEX, TIMES, LR)

    PARAMS_PATH = os.path.join(OUTPUT_PATH, PARAMS_NAME)
    PARAMS_PATH1 = os.path.join(OUTPUT_PATH, PARAMS_NAME1)
    LOG_PATH = os.path.join(OUTPUT_PATH, LOG_NAME)

    PARAMS_INIT_PATH = os.path.join(OUTPUT_PATH, PARAMS_INIT_NAME)

    setLogger(LOG_PATH, mode='w')

    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    train_dataset = MyDataset(TRAIN_INDEX_PATH, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, train_transform)
    valid_dataset = MyDataset(VALID_INDEX_PATH, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, eval_transform)
    test_dataset = MyDataset(TEST_INDEX_PATH, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    model = CANet().to(device)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.apply(initWeights)

    params = model.parameters()

    params_wd, params_rest = [], []
    for param_item in params:
        if param_item.requires_grad:
            (params_wd if param_item.dim() != 1 else params_rest).append(param_item)

    param_groups = [{'params': params_wd, 'weight_decay': WEIGHT_DECAY},
                    {'params': params_rest}]

    optimizer = optim.SGD(param_groups, lr=LR, momentum=0.9)

    EPOCHS = 230
    DECAY_EPOCH = [80, 130, 170]
    TMP = 170
    
    if statePath:
        logging.info('-' * 8)
        logging.info('Load state_dict in {}'.format(statePath))
        logging.info('-' * 8)

        all_state = torch.load(statePath)

        original_state = all_state['original_state']
        optimizer_state = all_state['optimizer_state']
        epoch = all_state['epoch']

        model.load_state_dict(original_state)
        optimizer.load_state_dict(optimizer_state)

        startEpoch = epoch + 1

    else:
        startEpoch = 1

    if LOAD_RATE != 0.5:
        all_state = torch.load(PARAMS_INIT_PATH)
        original_state = all_state['original_state']
        model.load_state_dict(original_state)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCH, gamma=0.1)
    best_acc = 0.0
    for epoch in range(startEpoch, EPOCHS + 1):
        scheduler.step()

        train(model, device, train_loader, optimizer, epoch)

        if epoch % EVAL_PRINT_FREQUENCY == 0:
            adjust_bn_stats(model, device, train_loader)
            best_acc = evaluate(model, device, valid_loader, epoch, optimizer, best_acc, PARAMS_PATH, PARAMS_PATH1,TMP)

    logging.info('\nTest set accuracy: \n')

    # load best parmater to test
    all_state = torch.load(PARAMS_PATH)
    original_state = all_state['original_state']
    optimizer_state = all_state['optimizer_state']
    model.load_state_dict(original_state)
    optimizer.load_state_dict(optimizer_state)

    adjust_bn_stats(model, device, train_loader)
    evaluate(model, device, test_loader, epoch, optimizer, best_acc, PARAMS_PATH, PARAMS_PATH1,TMP)


def myParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--DATASET_INDEX',
        help='Path for loading dataset',
        type=str,
        default=''
    )

    parser.add_argument(
        '-alg',
        '--STEGANOGRAPHY',
        help='embedding_algorithm',
        type=str,
        choices=['HILL-CMDC', 'SUNIWARD-CMDC','HILL-GINA', 'SUNIWARD-GINA','HILL-ACMP', 'SUNIWARD-ACMP'],
        required=True
    )

    parser.add_argument(
        '-rate',
        '--EMBEDDING_RATE',
        help='embedding_rate',
        type=str,
        choices=[ '0.2', '0.3', '0.4'],
        required=True
    )

    parser.add_argument(
        '-g',
        '--gpuNum',
        help='Determine which gpu to use',
        type=str,
        choices=['0', '1', '2', '3'],
        required=True
    )

    parser.add_argument(
        '-t',
        '--times',
        help='Determine which gpu to use',
        type=str,
        # required=True
        default=''
    )

    parser.add_argument(
        '-l',
        '--statePath',
        help='Path for loading model state',
        type=str,
        default=''
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = myParseArgs()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuNum
    main(args)
