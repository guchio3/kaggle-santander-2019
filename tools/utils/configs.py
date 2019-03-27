import yaml

from .logs import sel_log


def load_configs(path, logger=None):
    '''
    Load config file written in yaml format.

    '''
    with open(path, 'r') as fin:
        configs = yaml.load(fin)
    sel_log(f'USED_CONFIG: {path}', logger)
    sel_log(f'configs: {configs}', logger)
    return configs
