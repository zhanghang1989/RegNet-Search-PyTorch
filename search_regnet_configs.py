##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import copy
import pickle
import argparse
import configparser
from tqdm import tqdm

import torch
#torch.multiprocessing.set_start_method('spawn')
try:
    torch.multiprocessing.set_start_method('spawn',force=True)
except RuntimeError:
    pass
import torch.nn as nn
from torchvision import transforms

from thop import profile, clever_format
from regnet import config_regnet
from config_generator import GenConfg

import autotorch as at
from autotorch.core import Task
from autotorch.scheduler.resource import Resources
from autotorch.legacy import TaskScheduler

import encoding
from encoding.utils import (accuracy, AverageMeter, LR_Scheduler)

def get_args():
    # data settings
    parser = argparse.ArgumentParser(description='RegNet-AutoTorch')
    # config files
    parser.add_argument('--config-file-folder', type=str, required=True,
                        help='network model type (default: densenet)')
    # input size
    parser.add_argument('--crop-size', type=int, default=224,
                        help='crop image size')
    parser.add_argument('--base-size', type=int, default=None,
                        help='base image size')
    # data
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=25,
                        help='number of epochs to train (default: 600)')
    parser.add_argument('--workers', type=int, default=8,
                        help='dataloader threads')
    parser.add_argument('--data-dir', type=str, default=os.path.expanduser('~/.encoding/data'),
                        help='data location for training')
    # training hp
    parser.add_argument('--lr', type=float, default=0.1,
                            help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, 
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-5, 
                        help='SGD weight decay (default: 1e-4)')
    # AutoTorch
    parser.add_argument('--remote-file', type=str, default=None,
                        help='file to store remote ip addresses (default: None)')
    parser.add_argument('--checkname', type=str, default='checkpoint.ag',
                        help='checkpoint path (default: None)')
    parser.add_argument('--resume', action='store_true', default= False,
                        help='resume from the checkpoint if needed')
    parser = parser

    args = parser.parse_args()
    return args


def write_accuracy(config_file, acc):
    config = configparser.ConfigParser()
    config.read(config_file)
    config['net']['accuracy'] = str(acc)
    with open(config_file, 'w') as cfg:
        config.write(cfg)


def train_network(args, gpu, config_file):
    print('args:', args)
    # single gpu training only for evaluating the configurations
    model = config_regnet(config_file)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    model.cuda(gpu)
    criterion.cuda(gpu)

    # init dataloader
    base_size = args.base_size if args.base_size is not None else int(1.0 * args.crop_size / 0.875)
    transform = transforms.Compose([
            transforms.Resize(base_size),
            transforms.CenterCrop(args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    trainset = encoding.datasets.get_dataset('imagenet', root=args.data_dir,
                                             transform=transform, train=True, download=True)
    valset = encoding.datasets.get_dataset('imagenet', root=args.data_dir,
                                           transform=transform, train=False, download=True)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # lr scheduler
    lr_scheduler = LR_Scheduler('cos',
                                base_lr=args.lr,
                                num_epochs=args.epochs,
                                iters_per_epoch=len(train_loader))

    # write results into config file
    def train(epoch):
        model.train()
        top1 = AverageMeter()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            lr_scheduler(optimizer, batch_idx, epoch, 0)
            data, target = data.cuda(gpu), target.cuda(gpu)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    def validate():
        model.eval()
        top1 = AverageMeter()
        for batch_idx, (data, target) in enumerate(tqdm(val_loader)):
            data, target = data.cuda(gpu), target.cuda(gpu)
            with torch.no_grad():
                output = model(data)
                acc1 = accuracy(output, target, topk=(1,))
                top1.update(acc1[0], data.size(0))

        return top1.avg

    for epoch in range(0, args.epochs):
        train(epoch)
    acc = validate()
    write_accuracy(config_file, acc)


def get_config_files(folder):
    # find all config files in the folder
    files = []
    for filename in os.listdir(folder):
        if filename.endswith(".ini"):
            files.append(os.path.join(folder, filename))
    return files

def main():
    args = get_args()

    config_files = get_config_files(args.config_file_folder)

    scheduler = TaskScheduler()
    for i, config_file in enumerate(config_files):
        resource = Resources(num_cpus=8, num_gpus=1)
        task = Task(train_network, {
                'args': args,
                'gpu': i % at.get_gpu_count(),
                'config_file': config_file
            }, resource)
        scheduler.add_task(task)
        #train_network(args, 0, config_file)

    scheduler.join_tasks()
    at.save(scheduler.state_dict(), args.checkname)

if __name__ == '__main__':
    main()
