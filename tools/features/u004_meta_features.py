import os

import numpy as np
import pandas as pd

from .f015_f014_stats import f015_f014_stats
from .f016_nan_counts import f016_nan_counts
from .f019_filtered_round_features import f019_filtered_round_features

from ..utils.configs import load_configs
from ..utils.features import load_features
from ..utils.logs import dec_timer, sel_log


def _meta_features(df, feature_ids):
    _features = []
    if 'f015' in feature_ids:
        _features.append(f015_f014_stats(df))
    if 'f016' in feature_ids:
        _features.append(f016_nan_counts(df))
    if 'f019' in feature_ids:
        _features.append(f019_filtered_round_features(df))
    # merge cols
    # reset index to get id as a column
    features = pd.concat(_features, axis=1).reset_index()
    return features


@dec_timer
def _load_meta_features(
        feature_ids, trn_tst_df, trn_df, tst_df, logger):
    target_ids = [
        'f015',
        'f016',
        'f019',
    ]
    if len(set(target_ids) & set(feature_ids)) < 1:
        sel_log(f'''
                ======== {__name__} ========
                Stop feature making because even 1 element in exp_ids
                    {feature_ids}
                does not in target_ids
                    {target_ids}''', logger)
        return None, None, None

    sel_log(f'creating trn_tst_df ...', None)
    features = []
    for feature_id in feature_ids:
        configs = load_configs(f'./configs/c{feature_id}.yml', logger)
        features += configs['features']
    features = sorted(np.unique(features).tolist())
    trn_tst_df = load_features(
        features,
        './mnt/inputs/features/',
        os.cpu_count(),
        logger)

    return trn_tst_df, None, None
