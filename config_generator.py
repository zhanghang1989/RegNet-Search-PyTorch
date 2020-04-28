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
import configparser
from tqdm import tqdm

import torch
import autotorch as at

from thop import profile, clever_format
from regnet import config_regnet

def get_args():
    # data settings
    parser = argparse.ArgumentParser(description='RegNet-AutoTorch')
    # config files
    parser.add_argument('--config-file', type=str,
                        help='network model type (default: densenet)')
    # input size
    parser.add_argument('--crop-size', type=int, default=224,
                        help='crop image size')
    # target flops
    parser.add_argument('--gflops', type=float,
                         help='expected flops')
    # num configs
    parser.add_argument('--num-cofigs', type=int, default=32,
                        help='num of expected configs')
    parser = parser

    args = parser.parse_args()
    return args


@at.obj()
class GenConfg(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and \
                    not k in ('dump_config', 'update', 'pop'):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super().__setattr__(name, value)
        super().__setitem__(name, value)

    __setitem__ = __setattr__

    def dump_config(self, config_file=None):
        config = configparser.ConfigParser()
        config = configparser.ConfigParser()
        config['DEFAULT'] = {'bottleneck_ratio': '1'}
        config['net'] = {}
        for k, v in self.items():
            config['net'][k] = str(v)
        if config_file is not None:
            with open(config_file, 'w') as cfg:
                config.write(cfg)
        return config

def is_config_valid(cfg, target_flops, input_tensor, eps=1e-2):
    model = config_regnet(cfg.dump_config())
    flops, _ = profile(model, inputs=(input_tensor, ))
    #print(f"flops: {flops/1e9}")
    return flops <= (1. + eps) * target_flops and \
        flops >= (1. - eps) * target_flops

def main():
    args = get_args()

    input_tensor = torch.rand(1, 3, args.crop_size, args.crop_size)
    cfg_generator = GenConfg(
        initial_width=at.Int(16, 320),
        slope=at.Int(8, 96),
        quantized_param=at.Real(2.0, 3.2),
        network_depth=at.Int(12, 28),
        bottleneck_ratio=1,
        group_width=at.Int(8, 240))

    valid = 0
    while valid <= args.num_configs:
        cfg = cfg_generator.rand
        if is_config_valid(cfg, args.gflops*1e9, input_tensor):
            print(cfg)
            valid += 1
            cfg.dump_config(f'{args.config_file}-{valid}.ini')

if __name__ == '__main__':
    main()
