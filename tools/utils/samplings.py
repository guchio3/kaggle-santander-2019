import numpy as np
import pandas as pd

from ..utils.logs import sel_log


def get_neg_us_index(target, random_state=None):
    '''
    Get negative undersamling index.
    Positive and negative is not important here, so it can be more generalized.
    '''
    res_index = target[target == 1].index
    res_index = res_index.append(target[target == 0].sample(
        len(res_index), random_state=random_state).index)
    return res_index


def get_pos_os_index(target, os_lim, random_state=None):
    '''
    Get positive oversamling index.
    Positive and negative is not important here, so it can be more generalized.
    '''
    pos_size = target[target == 1].count()
    neg_size = target[target == 0].count()
    assert neg_size >= pos_size, \
        f'The sample of pos is more than neg ! ({pos_size} : {neg_size})'

    os_num = min(neg_size - pos_size, os_lim)
    res_index = target.index
    res_index = res_index.append(target[target == 1].sample(
        os_num, replace=True, random_state=random_state).index)
    return res_index


def resampling(target, resampling_type, random_state,
               os_lim=np.inf, logger=None):
    if resampling_type == 'under':
        sel_log('now under sampling ...', None)
        resampled_index = get_neg_us_index(target, random_state)
    elif resampling_type == 'over':
        sel_log('now over sampling ...', None)
        resampled_index = get_pos_os_index(target, os_lim, random_state)
    else:
        sel_log(f'ERROR: wrong resampling type ({resampling_type})', logger)
        sel_log('plz specify "under" or "over".', logger)
    return resampled_index
