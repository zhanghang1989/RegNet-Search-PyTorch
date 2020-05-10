# Search-RegNet-PyTorch

Reimplementing neural architecture search for RegNet using PyTorch and [AutoTorch](http://autotorch.org/).

| model| ref | Acc | config |
|------|------|------|---|
| RegNet-0.4GF| official | 72.38 | [link](./configs/RegNetX-0.4GF.ini) |
| RegNet-0.4GF| ours | 72.18 | [link](./gen_configs/ResNet-0.4GF/ResNet-0.4GF-1.ini) |
| RegNet-4.0GF| official | 79.03 | [link](./configs/RegNetX-4.0GF.ini) |

`official`: using official configuration. `ours`: using our searched configuration.

Training HP setting:
``
learning rate: 0.2,
batch size: 512,
weight decay: 1e-4,
``


## Quick Start

### Install Dependencies

- Install PyTorch, following the [instruction](https://pytorch.org/get-started/locally/).
- Install other dependencies:

```bash
pip install autotorch thop torch-encoding
```

- Install Apex (optional):

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
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
python train.py --dataset imagenet --config-file configs/RegNetX-4.0GF.ini --lr-scheduler cos --epochs 120 --checkname default --lr 0.025 --batch-size 64 --amp
```

## Architecture Search

### Generate config files with expected GFLOPs
```bash
python generate_configs.py --gflops 4 --num-configs 32 --config-file configs/RegNetX-4.0GF
```

The generated configuration files will be saved as `configs/RegNetX-4.0GF-1.ini`,
`configs/RegNetX-4.0GF-2.ini` ...

### Search best model for the config files in a folder
In this example, each model will be trained using a single gpu for 25 epochs. 

```bash
python search.py --config-file-folder gen_configs/RegNet-0.4GF/ --output-folder out_configs/ --epochs 25
```
The accuracy will be written into the output config file after training.

