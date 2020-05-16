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
import importlib
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

import autotorch as at

try:
    import apex
    from apex import amp
except ModuleNotFoundError:
    print('please install amp if using float16 training')

import encoding
from encoding.utils import (mkdir, accuracy, AverageMeter, LR_Scheduler)

def get_args():
    # data settings
    parser = argparse.ArgumentParser(description='RegNet-AutoTorch')
    # config files
    parser.add_argument('--arch', type=str, default='regnet',
                        help='network type (default: regnet)')
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
    parser.add_argument('--workers', type=int, default=12,
                        help='dataloader threads')
    parser.add_argument('--data-dir', type=str, default=os.path.expanduser('~/.encoding/data'),
                        help='data location for training')
    # training hp
    parser.add_argument('--amp', action='store_true',
                        default=False, help='using amp')
    parser.add_argument('--lr', type=float, default=0.1,
                            help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, 
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=5e-5, 
                        help='SGD weight decay (default: 1e-4)')
    # AutoTorch
    #parser.add_argument('--remote-file', type=str, default=None,
    #                    help='file to store remote ip addresses (default: None)')
    parser.add_argument('--num-trials', default=32, type=int,
                        help='number of trail tasks')
    parser.add_argument('--checkname', type=str, default='./exp/checkpoint.ag',
                        help='checkpoint path (default: None)')
    parser.add_argument('--resume', action='store_true', default= False,
                        help='resume from the checkpoint if needed')
    parser = parser

    args = parser.parse_args()
    return args


def write_results(cfg, out_config_file, **kwargs):
    config = configparser.ConfigParser()
    config['net'] = {}
    #config.read(in_config_file)
    for k, v in cfg.items():
        config['net'][k] = str(v)
    for k, v in kwargs.items():
        config['net'][k] = str(v)
    with open(out_config_file, 'w') as cfg:
        config.write(cfg)


@at.args()
def train_network(args, reporter):
    gpu = args.gpu_ids[0]
    print('gpu: {}, cfg: {}'.format(gpu, args.cfg))

    # single gpu training only for evaluating the configurations
    arch = importlib.import_module('arch.' + args.arch)
    from arch.base_generator import BaseGen
    #assert isinstance(args.cfg, BaseGen)
    model = arch.config_network(arch.dump_config(args.cfg))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.wd)

    model.cuda(gpu)
    criterion.cuda(gpu)

    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

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
    #from utils import SplitSampler
    #toy_sampler_train = SplitSampler(len(trainset), 200, 0)
    #toy_sampler_val = SplitSampler(len(valset), 200, 0)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,#sampler=toy_sampler_train,#
        num_workers=args.workers, drop_last=True, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=False,#sampler=toy_sampler_val, #
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
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
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

    out_config_file = os.path.join(args.output_folder, '{}.ini'.format(args.task_id))
    write_results(args.cfg, out_config_file,
                  accuracy=acc.item(), epochs=args.epochs,
                  lr=args.lr, wd=args.wd)
    reporter(epoch=args.epochs-1, accuracy=float(acc.item()/100.0))


def load_configs(folder, args, prefix=None, overwrite=False):
    #arch = importlib.import_module('arch.' + args.arch)
    from arch.base_generator import BaseGen
    def is_trained(cfg):
        # check if this config has been trained
        if 'accuracy' in cfg.keys() and cfg['accuracy'] > 0 and \
                cfg['epochs'] == args.epochs:
            return True
        return False
    def add_prefix(cfg, prefix):
        new_cfg = {}
        if prefix is not None:
            for k, v in cfg.items():
                new_cfg[prefix + '.' + k] = v
            return new_cfg
        else:
            return cfg
    # find all config files in the folder
    configs = []
    for filename in os.listdir(folder):
        if filename.endswith(".ini"):
            fullname = os.path.join(folder, filename)
            config = BaseGen()
            config.load_config(fullname)
            if not overwrite and is_trained(config): continue
            configs.append(add_prefix(config, prefix))
    return configs


def main():
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    #logging.basicConfig(level=logging.DEBUG)

    args = train_network.args
    args.update(**vars(get_args()))

    #args = get_args()
    mkdir(args.output_folder)

    configs = load_configs(args.config_file_folder, args, 'cfg')
    print(f"len(configs): {len(configs)}")

    arch = importlib.import_module('arch.' + args.arch)
    train_network.update(cfg=arch.GenConfg())

    import autotorch as at

    searcher = at.searcher.BayesOptSearcher(train_network.cs, lazy_configs=configs)
    myscheduler = at.scheduler.FIFOScheduler(train_network, args,
                                searcher=searcher,
                                resource={'num_cpus': args.workers,
                                          'num_gpus': 1},
                                num_trials=args.num_trials,
                                checkpoint=args.checkname,
                                resume=args.resume,
                                time_attr='epoch',
                                reward_attr="accuracy")

    print(myscheduler)
    myscheduler.run()
    myscheduler.join_jobs()

    myscheduler.get_training_curves('{}.png'.format(os.path.splitext(args.checkname)[0])) 
    print('The Best Configuration and Accuracy are: {}, {}'.format(myscheduler.get_best_config(),
                                                                   myscheduler.get_best_reward()))


if __name__ == '__main__':
    main()

