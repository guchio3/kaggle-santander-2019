import numpy as np
import pandas as pd
from numba import jit

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
    if resampling_type == 'none':
        sel_log('none, which means not apply resampling !', logger)
        resampled_index = target.index
    elif resampling_type == 'b_under':
        sel_log('now binary under sampling ...', logger)
        resampled_index = get_binary_us_index(target, random_state)
    elif resampling_type == 'b_over':
        sel_log('now binary over sampling ...', logger)
        resampled_index = get_binary_os_index(target, os_lim, random_state)
    else:
        sel_log(f'ERROR: wrong resampling type ({resampling_type})', logger)
        sel_log('plz specify "under" or "over".', logger)
    return resampled_index


def get_binary_us_values(features_df, target, random_state=None):
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
    res_features, res_target = features_df.iloc[res_index], target[res_index]
    return res_features, res_target


def get_binary_os_values(features_df, target, os_lim, random_state=None):
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
    res_features, res_target = features_df.iloc[res_index], target[res_index]
    return res_features, res_target


@jit
def get_binary_random_augment_values(
        features_df, target, pos_t, neg_t, random_state=71):
    pos_ids = target[target == 1].index
    neg_ids = target[target == 0].index
    res_dfs = [features_df]
    res_targets = [target]
    # augment positive values
    for i in range(pos_t):
        pos_df = features_df.loc[pos_ids].copy()
        for j, col in enumerate(pos_df.columns):
            pos_df[col] = pos_df[col].sample(
                frac=1, random_state=random_state * i + j)
        res_dfs.append(pos_df)
        res_targets.append(pd.Series(np.ones(pos_df.shape[0])))
    # augment negative values
    for i in range(neg_t):
        neg_df = features_df.loc[neg_ids].copy()
        for j, col in enumerate(neg_df.columns):
            neg_df[col] = neg_df[col].sample(
                frac=1, random_state=random_state * i + j)
        res_dfs.append(neg_df)
        res_targets.append(pd.Series(np.zeros(neg_df.shape[0])))
    # concat values
    res_features = pd.concat(res_dfs, axis=0).reset_index(drop=True)
    res_target = pd.concat(res_targets, axis=0).reset_index(drop=True)
    return res_features, res_target


@jit
def get_binary_random_augment_values_w_pairs(
        features_df, target, pos_t, neg_t, random_state=71):
    pos_ids = target[target == 1].index
    neg_ids = target[target == 0].index
    res_dfs = [features_df]
    res_targets = [target]
    # augment positive values
    for i in range(pos_t):
        pos_df = features_df.loc[pos_ids].copy()
        for j in range(pos_df.shape[1] - 600):
            k = j + 600 if j > 599 else [j, j + 200, j + 400, j + 600]
            pos_df.iloc[:, k] = pos_df.iloc[:, k].sample(
                frac=1, random_state=random_state * i * 10 + j).values
        res_dfs.append(pos_df)
        res_targets.append(pd.Series(np.ones(pos_df.shape[0])))
    # augment negative values
    for i in range(neg_t):
        neg_df = features_df.loc[neg_ids].copy()
        for j in range(neg_df.shape[1] - 600):
            k = j + 600 if j > 599 else [j, j + 200, j + 400, j + 600]
            neg_df.iloc[:, k] = neg_df.iloc[:, k].sample(
                frac=1, random_state=random_state * i * 10 + j).values
        res_dfs.append(neg_df)
        res_targets.append(pd.Series(np.zeros(neg_df.shape[0])))
    # concat values
    res_features = pd.concat(res_dfs, axis=0).reset_index(drop=True)
    res_target = pd.concat(res_targets, axis=0).reset_index(drop=True)
    return res_features, res_target


@jit
def get_binary_random_augment_values_w_pairs_2(
        features_df, target, pos_t, neg_t, random_state=71):
    pos_ids = target[target == 1].index
    neg_ids = target[target == 0].index
    res_dfs = [features_df]
    res_targets = [target]
    # augment positive values
    for i in range(pos_t):
        pos_df = features_df.loc[pos_ids].copy()
        for j in range(pos_df.shape[1] - 800):
            k = j + 800 if j > 799 else [j, j + 200, j + 400, j + 600, j + 800]
            pos_df.iloc[:, k] = pos_df.iloc[:, k].sample(
                frac=1, random_state=random_state * i * 10 + j).values
        res_dfs.append(pos_df)
        res_targets.append(pd.Series(np.ones(pos_df.shape[0])))
    # augment negative values
    for i in range(neg_t):
        neg_df = features_df.loc[neg_ids].copy()
        for j in range(neg_df.shape[1] - 800):
            k = j + 800 if j > 799 else [j, j + 200, j + 400, j + 600, j + 800]
            neg_df.iloc[:, k] = neg_df.iloc[:, k].sample(
                frac=1, random_state=random_state * i * 10 + j).values
        res_dfs.append(neg_df)
        res_targets.append(pd.Series(np.zeros(neg_df.shape[0])))
    # concat values
    res_features = pd.concat(res_dfs, axis=0).reset_index(drop=True)
    res_target = pd.concat(res_targets, axis=0).reset_index(drop=True)
    return res_features, res_target


def value_resampling(features_df, target, resampling_type, random_state,
                     os_lim=np.inf, pos_t=2, neg_t=1, logger=None):
    if resampling_type == 'none':
        sel_log('none, which means not apply resampling !', logger)
        res_features, res_target = features_df, target
    elif resampling_type == 'b_under':
        sel_log('now binary under sampling ...', logger)
        res_features, res_target = get_binary_us_values(
            features_df, target, random_state)
    elif resampling_type == 'b_over':
        sel_log('now binary over sampling ...', logger)
        res_features, res_target = get_binary_os_values(
            features_df, target, os_lim, random_state)
    elif resampling_type == 'b_rand_aug':
        sel_log('now random augmentation sampling ...', logger)
        res_features, res_target = get_binary_random_augment_values(
            features_df, target, pos_t, neg_t, random_state=random_state)
    elif resampling_type == 'b_rand_aug_pair':
        sel_log('now random augmentation sampling w/ pairs ...', logger)
        res_features, res_target = get_binary_random_augment_values_w_pairs(
            features_df, target, pos_t, neg_t, random_state=random_state)
    elif resampling_type == 'b_rand_aug_pair_2':
        sel_log('now random augmentation sampling w/ pairs ...', logger)
        res_features, res_target = get_binary_random_augment_values_w_pairs_2(
            features_df, target, pos_t, neg_t, random_state=random_state)
    else:
        sel_log(f'ERROR: wrong resampling type ({resampling_type})', logger)
        sel_log('plz specify "under" or "over".', logger)
    return res_features, res_target
