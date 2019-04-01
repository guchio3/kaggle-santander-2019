import os

import numpy as np
import pandas as pd

from ..utils.configs import load_configs
from ..utils.features import load_features
from ..utils.logs import dec_timer, sel_log
from .f007_uniq_cnt_meta_features import f007_uniq_cnt_meta_features
from .f017_filtered_uniq_cnt_encoding import f017_filtered_uniq_cnt_encoding
from .f018_filtered_uniq_cnt_encoding_real import \
    f018_filtered_uniq_cnt_encoding_real
from .f023_standard_scale_rowwise_stats_features import \
    f023_standard_scale_rowwise_stats_features


def _colwise_meta_features(df, feature_ids):
    _features = []
    if 'f007' in feature_ids:
        _features.append(f007_uniq_cnt_meta_features(df))
    if 'f017' in feature_ids:
        _features.append(f017_filtered_uniq_cnt_encoding(df))
    if 'f018' in feature_ids:
        _features.append(f018_filtered_uniq_cnt_encoding_real(df))
    if 'f023' in feature_ids:
        _features.append(f023_standard_scale_rowwise_stats_features(df))
    features = pd.concat(_features, axis=1)
    return features


@dec_timer
def _load_colwise_meta_features(
        feature_ids, trn_tst_df, trn_df, tst_df, logger):
    target_ids = [
        'f007',
        'f017',
        'f018',
        'f023',
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
