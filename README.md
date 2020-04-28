# Search-RegNet-PyTorch
Search for RegNet using PyTorch


## Quick Start
### Test model #params and FLOPs from config file
```bash
python test_flops.py --config-file configs/RegNetX-4.0GF.ini
```

### Train model from a config file
```bash
TODO
```

### Generate config files with expected GFLOPs
```bash
python config_generator.py --gflops 4 --num-configs 32 --config-file configs/RegNetX-4.0GF
```

### Search best model for a certain GFLOPs
```bash
TODO
```