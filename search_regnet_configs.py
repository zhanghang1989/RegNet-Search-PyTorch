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
import random
import pickle
import logging
import argparse
import configparser
from tqdm import tqdm

import torch
import multiprocessing as mp
import multiprocessing.pool
try:
    torch.multiprocessing.set_start_method('spawn',force=True)
except RuntimeError:
    pass
import torch.nn as nn
from torchvision import transforms

from thop import profile, clever_format
from regnet import config_regnet
from config_generator import GenConfg

import encoding
from encoding.utils import (mkdir, accuracy, AverageMeter, LR_Scheduler)

def get_args():
    # data settings
    parser = argparse.ArgumentParser(description='RegNet-AutoTorch')
    # config files
    parser.add_argument('--config-file-folder', type=str, required=True,
                        help='network model type (default: densenet)')
    parser.add_argument('--output-folder', type=str, required=True,
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
    parser.add_argument('--wd', type=float, default=5e-5, 
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


def write_accuracy(in_config_file, out_config_file, **kwargs):
    config = configparser.ConfigParser()
    config.read(in_config_file)
    for k, v in kwargs.items():
        config['net'][k] = str(v)
    with open(out_config_file, 'w') as cfg:
        config.write(cfg)

class SplitSampler(torch.utils.data.Sampler):
    """ Split the dataset into `num_parts` parts and sample from the part with
    index `part_index`
 
    Parameters
    ----------
    length: int
      Number of examples in the dataset
    num_parts: int
      Partition the data into multiple parts
    part_index: int
      The index of the part to read from
    """
    def __init__(self, length, num_parts=1, part_index=0, random=True):
        # Compute the length of each partition
        self.part_len = length // num_parts
        # Compute the start index for this partition
        self.start = self.part_len * part_index
        # Compute the end index for this partition
        self.end = self.start + self.part_len
        self.random = random
 
    def __iter__(self):
        # Extract examples between `start` and `end`, shuffle and return them.
        indices = list(range(self.start, self.end))
        if self.random:
            random.shuffle(indices)
        return iter(indices)
 
    def __len__(self):
        return self.part_len

def train_network(args, gpu_manager, config_file):
    gpu = gpu_manager.request()
    print('gpu: {}, cfg: {}'.format(gpu, config_file))

    # single gpu training only for evaluating the configurations
    model = config_regnet(config_file)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.wd)

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
    #toy_sampler = SplitSampler(len(trainset), 100)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,#sampler=toy_sampler,#
        num_workers=args.workers, drop_last=True, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # lr scheduler
    lr_scheduler = LR_Scheduler('cos',
                                base_lr=args.lr,
                                num_epochs=args.epochs,
                                iters_per_epoch=len(train_loader),
                                quiet=True)

    # write results into config file
    def train(epoch):
        model.train()
        top1 = AverageMeter()
        for batch_idx, (data, target) in enumerate(train_loader):
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
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.cuda(gpu), target.cuda(gpu)
            with torch.no_grad():
                output = model(data)
                acc1 = accuracy(output, target, topk=(1,))
                top1.update(acc1[0], data.size(0))

        return top1.avg

    for epoch in tqdm(range(0, args.epochs)):
        train(epoch)
    acc = validate()

    out_config_file = os.path.join(args.output_folder, os.path.basename(config_file))
    write_accuracy(config_file, out_config_file,
                   accuracy=acc.item(), epochs=args.epochs,
                   lr=args.lr, wd=args.wd)
    gpu_manager.release(gpu)


def get_config_files(folder, overwrite=True):
    def is_trained(filename):
        # check if this config has been trained
        return False
    # find all config files in the folder
    files = []
    for filename in os.listdir(folder):
        if filename.endswith(".ini"):
            fullname = os.path.join(folder, filename)
            if not overwrite and is_trained(fullname): continue
            files.append(fullname)
    return files

def train_network_map(args):
    train_network(*args)

class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class MyPool(mp.pool.Pool):
    Process = NoDaemonProcess


class GPUManager(object):
    def __init__(self, ngpus):
        self._gpus = mp.Manager().Queue()
        for i in range(ngpus):
            self._gpus.put(i)

    def request(self):
        return self._gpus.get()

    def release(self, gpu):
        self._gpus.put(gpu)

def main():
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    logging.basicConfig(level=logging.DEBUG)

    args = get_args()
    mkdir(args.output_folder)

    config_files = get_config_files(args.config_file_folder)
    print(f"len(config_files): {len(config_files)}")

    ngpus = torch.cuda.device_count()
    gpu_manager = GPUManager(ngpus)
    tasks = ([args, gpu_manager, config_file] for i, config_file in enumerate(config_files))
        
    p = MyPool(processes=ngpus)
    p.map(train_network_map, tasks)

if __name__ == '__main__':
    main()
