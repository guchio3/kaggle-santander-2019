import pandas as pd

from ..utils.logs import dec_timer, sel_log

from .f004_mode_shift_pow_features import f004_mode_shift_pow_features
from .f006_uniq_cnt_encoding_features import f006_uniq_cnt_encoding_features
from .f009_uniq_mask_features import f009_uniq_mask_features
from .f010_non_uniq_features import f010_non_uniq_features
from .f011_only_trn_non_uniq_features import f011_only_trn_non_uniq_features
from .f012_common_features import f012_common_features
from .f014_non_uniq_real_features import f014_non_uniq_real_features
from .f020_f014_2uniq import f020_f014_2uniq
from .f021_uniq_real_features import f021_uniq_real_features
from .f022_2uniq_real_features import f022_2uniq_real_features
from .f024_col_each_cnt_raw_features import f024_col_each_cnt_raw_features
from .f025_f024_or_more_features import f025_f024_or_more_features
from .f026_f014_3uniq import f026_f014_3uniq
from .f027_f014_3nonuniq import f027_f014_3nonuniq
from .f028_f014_4uniq import f028_f014_4uniq
from .f029_f014_4nonuniq import f029_f014_4nonuniq


def _colwise_features(df, feature_ids):
    _features = []
    if 'f004' in feature_ids:
        _features.append(f004_mode_shift_pow_features(df))
    if 'f006' in feature_ids:
        _features.append(f006_uniq_cnt_encoding_features(df))
    if 'f009' in feature_ids:
        _features.append(f009_uniq_mask_features(df))
    if 'f010' in feature_ids:
        _features.append(f010_non_uniq_features(df))
    if 'f011' in feature_ids:
        _features.append(f011_only_trn_non_uniq_features(df))
    if 'f012' in feature_ids:
        _features.append(f012_common_features(df))
    if 'f014' in feature_ids:
        _features.append(f014_non_uniq_real_features(df))
    if 'f020' in feature_ids:
        _features.append(f020_f014_2uniq(df))
    if 'f021' in feature_ids:
        _features.append(f021_uniq_real_features(df))
    if 'f022' in feature_ids:
        _features.append(f022_2uniq_real_features(df))
    if 'f024' in feature_ids:
        _features.append(f024_col_each_cnt_raw_features(df))
    if 'f025' in feature_ids:
        _features.append(f025_f024_or_more_features(df))
    if 'f026' in feature_ids:
        _features.append(f026_f014_3uniq(df))
    if 'f027' in feature_ids:
        _features.append(f027_f014_3nonuniq(df))
    if 'f028' in feature_ids:
        _features.append(f028_f014_4uniq(df))
    if 'f029' in feature_ids:
        _features.append(f029_f014_4nonuniq(df))
    # merge cols
    # reset index to get id as a column
    features = pd.concat(_features, axis=1).reset_index()
    return features


@dec_timer
def _load_colwise_features(
        feature_ids, trn_tst_df, trn_df, tst_df, logger):
    target_ids = [
        'f004',
        'f006',
        'f009',
        'f010',
        'f011',
        'f012',
        'f014',
        'f020',
        'f021',
        'f022',
        'f024',
        'f025',
        'f026',
        'f027',
        'f028',
        'f029',
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
