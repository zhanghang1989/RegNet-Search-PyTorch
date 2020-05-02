# Search-RegNet-PyTorch

Search for RegNet using PyTorch and [AutoTorch](http://autotorch.org/).

## Quick Start

### Install Dependencies

- Install PyTorch, following the [instruction](https://pytorch.org/get-started/locally/).
- Install other dependencies:
```bash
pip install autotorch thop torch-encoding
```

### Test #Params and FLOPs from config file
```bash
python test_flops.py --config-file configs/RegNetX-4.0GF.ini
```

## Single Model Training

### Prepare ImageNet Dataset
```
cd scripts/
# assuming you have downloaded the dataset in the current folder
python prepare_imagenet.py --download-dir ./
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

The generated configuration files will be saved as `configs/RegNetX-4.0GF-1.ini`,
`configs/RegNetX-4.0GF-2.ini` ...

### Search best model for the config files in a folder
In this example, each model will be trained using a single gpu for 10 epochs. 

```bash
python search_regnet_configs.py --config-file-folder gen_configs/0.4GF/ --output-folder out_configs/ --epochs 25
```
The accuracy will be written into config file after training.
