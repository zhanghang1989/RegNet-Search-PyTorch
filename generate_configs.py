##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import argparse
import importlib
import configparser
from tqdm import tqdm

import torch
import autotorch as at

from thop import profile, clever_format

def get_args():
    # data settings
    parser = argparse.ArgumentParser(description='RegNet-AutoTorch')
    # config files
    parser.add_argument('--arch', type=str, default='regnet',
                        help='network type (default: regnet)')
    parser.add_argument('--config-file', type=str, required=True,
                        help='target config file prefix')
    # input size
    parser.add_argument('--crop-size', type=int, default=224,
                        help='crop image size')
    # target flops
    parser.add_argument('--gflops', type=float, required=True,
                         help='expected flops')
    parser.add_argument('--eps', type=float, default=2e-2,
                         help='eps for expected flops')
    # num configs
    parser.add_argument('--num-configs', type=int, default=32,
                        help='num of expected configs')
    parser = parser

    args = parser.parse_args()
    return args


def is_config_valid(arch, cfg, target_flops, input_tensor, eps):
    model = arch.config_network(cfg.dump_config())
    flops, _ = profile(model, inputs=(input_tensor, ))
    return flops <= (1. + eps) * target_flops and \
        flops >= (1. - eps) * target_flops

def main():
    args = get_args()

    input_tensor = torch.rand(1, 3, args.crop_size, args.crop_size)
    arch = importlib.import_module('arch.' + arch)
    cfg_generator = arch.GenConfg()

    valid = 0
    pbar = tqdm(range(args.num_configs))
    while valid < args.num_configs:
        cfg = cfg_generator.rand
        if is_config_valid(arch, cfg, args.gflops*1e9, input_tensor, args.eps):
            pbar.update()
            print(cfg)
            valid += 1
            cfg.dump_config(f'{args.config_file}-{valid}.ini')

if __name__ == '__main__':
    main()
