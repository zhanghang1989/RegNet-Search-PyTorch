##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import argparse
import importlib

import torch
from thop import profile, clever_format

def get_args():
    # data settings
    parser = argparse.ArgumentParser(description='RegNet-AutoTorch')
    # config files
    parser.add_argument('--arch', type=str, default='regnet',
                        help='network type (default: regnet)')
    parser.add_argument('--config-file', type=str, required=True,
                        help='network model type (default: densenet)')
    parser.add_argument('--display', action='store_true', default=False,
                        help='display network')
    # input size
    parser.add_argument('--crop-size', type=int, default=224,
                        help='crop image size')
    parser = parser

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    arch = importlib.import_module('arch.' + args.arch)
    model = arch.config_network(args.config_file)
    if args.display:
        print(model)

    dummy_images = torch.rand(1, 3, args.crop_size, args.crop_size)

    macs, params = profile(model, inputs=(dummy_images, ))
    macs, params = clever_format([macs, params], "%.3f") 

    print(f"macs: {macs}, params: {params}")

if __name__ == '__main__':
    main()

