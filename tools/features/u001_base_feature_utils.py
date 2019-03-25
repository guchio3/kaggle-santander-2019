import pandas as pd

from ..utils.logs import dec_timer, sel_log

from .f001_raw_features import f001_raw_features
from .f002_900_kernel_stat_features import f002_900_kernel_stat_features
from .f003_900_kernel_round_features import f003_900_kernel_round_features


def _base_features(df, feature_ids):
    _features = []
    if 'f001' in feature_ids:
        _features.append(f001_raw_features(df))
    if 'f002' in feature_ids:
        _features.append(f002_900_kernel_stat_features(df))
    if 'f003' in feature_ids:
        _features.append(f003_900_kernel_round_features(df))
    features = pd.concat(_features, axis=1)
    return features


@dec_timer
def _load_base_features(
        feature_ids, trn_tst_df, trn_df, tst_df, logger):
    target_ids = [
        'f001',
        'f002',
        'f003',
    ]
    if len(set(target_ids) & set(feature_ids)) < 1:
        sel_log(f'''
                ======== {__name__} ========
                Stop feature making because even 1 element in exp_ids
                    {feature_ids}
                does not in target_ids
                    {target_ids}''', logger)
        return None, None, None

    trn_path = './mnt/inputs/origin/train.csv.zip'
    tst_path = './mnt/inputs/origin/test.csv.zip'

    # Load dfs if not input.
    if trn_df is None:
        sel_log(f'loading {trn_path} ...', None)
        trn_df = pd.read_csv(trn_path, compression='zip')
    if tst_df is None:
        sel_log(f'loading {tst_path} ...', None)
        tst_df = pd.read_csv(tst_path, compression='zip')
    if trn_tst_df is None:
        sel_log(f'creating trn_tst_df ...', None)
        trn_tst_df = pd.concat([trn_df, tst_df], axis=0, sort=False)

    return trn_tst_df, trn_df, tst_df
