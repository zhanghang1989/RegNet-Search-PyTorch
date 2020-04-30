# Search-RegNet-PyTorch

Search for RegNet using PyTorch and [AutoTorch](http://autotorch.org/).

## Quick Start

### Install Dependencies

- Install PyTorch, following the [instruction](https://pytorch.org/get-started/locally/).
- Install other dependencies:
```bash
pip install autotorch thop torch-encoding
```

### Test #params and FLOPs from config file
```bash
python test_flops.py --config-file configs/RegNetX-4.0GF.ini
```

### Train a single model from a config file
```bash
TODO
```

## Architecture Search

### Generate config files with expected GFLOPs
```bash
python config_generator.py --gflops 4 --num-configs 32 --config-file configs/RegNetX-4.0GF
```

### Search best model for the configurations in a folder
```bash
python search_regnet_configs.py --config-file-folder gen_configs_04gf/ --epochs 10 --data-dir /media/ramdisk/
```
