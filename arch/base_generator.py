import configparser

class BaseGen(dict):
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
                    not k in ('dump_config', 'load_config', 'update', 'pop'):
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
        config['DEFAULT'] = {'bottleneck_ratio': '1'}
        config['net'] = {}

        for k, v in self.items():
            config['net'][k] = str(v)
        if config_file is not None:
            with open(config_file, 'w') as cfg:
                config.write(cfg)
        return config

    def load_config(self, filename):
        config = configparser.ConfigParser()
        config.read(filename)

        for k, v in config['net'].items():
            setattr(self, k, v)
