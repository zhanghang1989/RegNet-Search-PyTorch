import configparser
import numpy as np

import torch
import torch.nn as nn

import autotorch as at
from .base_generator import BaseGen

__all__ = ['RegNeSt']

# code modified from https://github.com/signatrix/regnet
class AnyNeSt(nn.Module):
    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width,
                 stride, se_ratio):
        super().__init__()
        for block_width, bottleneck_ratio, group_width in zip(ls_block_width, ls_bottleneck_ratio, ls_group_width):
            assert block_width % (bottleneck_ratio * group_width) == 0
        self.net = nn.Sequential()
        prev_block_width = 32
        self.net.add_module("stem", ConvBnAct(3, prev_block_width, kernel_size=3, stride=2, padding=1, bias=False))

        for i, (num_blocks, block_width, bottleneck_ratio, group_width) in \
                enumerate(zip(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width)):
            self.net.add_module("stage_{}".format(i),
                                Stage(num_blocks, prev_block_width, block_width,
                                      bottleneck_ratio, group_width=group_width,
                                      stride=stride, se_ratio=se_ratio))
            prev_block_width = block_width

        self.net.add_module("pool", GlobalAvgPool2d())
        self.net.add_module("fc", nn.Linear(ls_block_width[-1], 1000))

    def forward(self, x):
        x = self.net(x)
        return x


class RegNetX(AnyNeSt):
    def __init__(self, initial_width, slope, quantized_param, network_depth, bottleneck_ratio, group_width,
                 stride=2):
        # We need to derive block width and number of blocks from initial parameters.
        parameterized_width = initial_width + slope * np.arange(network_depth)
        parameterized_block = np.log(parameterized_width / initial_width) / np.log(quantized_param)
        parameterized_block = np.round(parameterized_block)
        quantized_width = initial_width * np.power(quantized_param, parameterized_block)
        # We need to convert quantized_width to make sure that it is divisible by 8
        quantized_width = 8 * np.round(quantized_width / 8)
        ls_block_width, ls_num_blocks = np.unique(quantized_width.astype(np.int), return_counts=True)
        # At this points, for each stage, the above-calculated block width could be incompatible to group width
        # due to bottleneck ratio. Hence, we need to adjust the formers.
        # Group width could be swapped to number of groups, since their multiplication is block width
        ls_group_width = np.array([min(group_width, block_width // bottleneck_ratio) for block_width in ls_block_width])
        ls_block_width = np.round(ls_block_width // bottleneck_ratio / group_width) * group_width
        ls_group_width = ls_group_width.astype(np.int) * bottleneck_ratio
        ls_bottleneck_ratio = [bottleneck_ratio for _ in range(len(ls_block_width))]
        ls_group_width  = ls_group_width.tolist()
        ls_block_width = ls_block_width.astype(np.int).tolist()

        super().__init__(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width,
                         stride=stride, se_ratio=None)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_ratio, group_width,
                 stride, se_ratio):
        super(Bottleneck, self).__init__()
        inter_channels = out_channels // bottleneck_ratio
        groups = inter_channels // group_width

        self.conv1 = ConvBnAct(in_channels, inter_channels, kernel_size=1, bias=False)
        self.conv2 = ConvBnAct(inter_channels, inter_channels, kernel_size=3, stride=stride,
                                      groups=groups, padding=1, bias=False)
        if se_ratio is not None:
            se_channels = in_channels // se_ratio
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Conv2d(inter_channels, se_channels, kernel_size=1, bias=True),
                nn.ReLU(),
                nn.Conv2d(se_channels, inter_channels, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.se = None
        self.conv3 = ConvBnAct(inter_channels, out_channels, kernel_size=1, bias=False, act=False)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBnAct(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, act=False)
        else:
            self.shortcut = None
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        if self.se is not None:
            x1 = x1 * self.se(x1)
        x1 = self.conv3(x1)
        if self.shortcut is not None:
            x2 = self.shortcut(x)
        else:
            x2 = x
        x = self.relu(x1 + x2)
        return x


class Stage(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, bottleneck_ratio, group_width, stride, se_ratio):
        super().__init__()
        self.blocks = nn.Sequential()
        self.blocks.add_module("block_0", Bottleneck(in_channels, out_channels, bottleneck_ratio, group_width,
                                                     stride=stride, se_ratio=se_ratio))
        for i in range(1, num_blocks):
            self.blocks.add_module("block_{}".format(i),
                                   Bottleneck(out_channels, out_channels, bottleneck_ratio, group_width,
                                              stride=1, se_ratio=se_ratio))

    def forward(self, x):
        x = self.blocks(x)
        return x


class ConvBnAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', act=True,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                          padding=padding, dilation=dilation, groups=groups, bias=bias,
                                          padding_mode=padding_mode))
        self.add_module("bn", nn.BatchNorm2d(out_channels))
        if act:
            self.add_module("relu", nn.ReLU())


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)

def dump_config(cfg, config_file=None):
    config = configparser.ConfigParser()
    config['DEFAULT'] = {'bottleneck_ratio': '1'}
    config['net'] = {}
    cfg.group_width = cfg.group_width if cfg.group_width <= cfg.initial_width \
        else cfg.initial_width
    cfg.group_width = int(cfg.group_width // 8 * 8)
    for k, v in cfg.items():
        config['net'][k] = str(v)
    if config_file is not None:
        with open(config_file, 'w') as cfg:
            config.write(cfg)
    return config

def convert_config_space(cfg):
    ncfg = GenConfg().init()
    ncfg.bottleneck_ratio = 1
    #ncfg.radix = int(cfg.radix)
    #ncfg.cardinality = int(cfg.cardinality)
    ncfg.slope = float(cfg.slope)
    ncfg.initial_width = int(cfg.initial_width)
    ncfg.network_depth = int(cfg.network_depth)
    ncfg.group_width = int(cfg.group_width)
    ncfg.quantized_param = float(cfg.quantized_param)
    acc = float(cfg.accuracy) if hasattr(cfg, 'accuracy') else 0.0
    return ncfg, acc

@at.obj(
    bottleneck_ratio=at.Int(1, 2),
    #initial_width=at.Int(16, 320),
    initial_width=at.Int(16, 96, log=True),
    #slope=at.Real(24, 128, log=True),
    slope=at.Real(24, 64, log=True),
    quantized_param=at.Real(2.0, 3.2),
    network_depth=at.Int(12, 28),
    #group_width=at.Int(8, 240),
    group_width=at.Int(8, 64),
)
class GenConfg(BaseGen):
    def dump_config(self, config_file=None):
        return dump_config(self, config_file)

def config_network(cfg):
    # construct regnet from a config file
    if isinstance(cfg, configparser.ConfigParser):
        config = cfg
    else:
        config = configparser.ConfigParser()
        config.read(cfg)
    bottleneck_ratio = int(config['net']['bottleneck_ratio'])
    group_width = int(config['net']['group_width'])
    initial_width = int(config['net']['initial_width'])

    group_width = group_width if group_width <= initial_width else initial_width
    group_width = int(group_width // 8 * 8)

    #initial_width = int(initial_width // group_width * group_width)

    slope = float(config['net']['slope'])
    quantized_param = float(config['net']['quantized_param'])

    network_depth = int(config['net']['network_depth'])
    model = RegNetX(initial_width, slope, quantized_param, network_depth,
                    bottleneck_ratio, group_width)
    return model
