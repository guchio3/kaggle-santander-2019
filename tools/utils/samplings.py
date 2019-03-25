import numpy as np
import pandas as pd

from ..utils.logs import sel_log


def get_binary_us_index(target, random_state=None):
    '''
    Get binary undersamling index.

    '''
    pos_size = target[target == 1].count()
    neg_size = target[target == 0].count()
    less_target = 1 if pos_size < neg_size else 0
    more_target = (less_target + 1) % 2

    res_index = target[target == less_target].index
    res_index = res_index.append(target[target == more_target].sample(
        len(res_index), random_state=random_state).index)
    return res_index


def get_binary_os_index(target, os_lim, random_state=None):
    '''
    Get binary oversamling index.

    '''
    pos_size = target[target == 1].count()
    neg_size = target[target == 0].count()
    less_target = 1 if pos_size < neg_size else 0

    os_num = min(abs(neg_size - pos_size), os_lim)
    res_index = target.index
    res_index = res_index.append(target[target == less_target].sample(
        os_num, replace=True, random_state=random_state).index)
    return res_index


def resampling(target, resampling_type, random_state,
               os_lim=np.inf, logger=None):
    if resampling_type == 'b_under':
        sel_log('now binary under sampling ...', logger)
        resampled_index = get_binary_us_index(target, random_state)
    elif resampling_type == 'b_over':
        sel_log('now binary over sampling ...', logger)
        resampled_index = get_binary_os_index(target, os_lim, random_state)
    else:
        sel_log(f'ERROR: wrong resampling type ({resampling_type})', logger)
        sel_log('plz specify "under" or "over".', logger)
    return resampled_index
