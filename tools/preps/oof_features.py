import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def mk_oof_features(oof_filename, sub_filename, col_name):
    oof_feature = pd.DataFrame()
    oof_feature['ID_code'] = pd.read_pickle(
        './mnt/inputs/features/ID_code.pkl.gz').values
    oof_feature = oof_feature.set_index('ID_code')
    oof_df = pd.read_csv(oof_filename)
    sub_df = pd.read_csv(sub_filename)
    oof_feature[col_name] = np.nan
    oof_feature.loc[oof_df.ID_code, col_name] = oof_df['oof_proba'].values
    oof_feature.loc[sub_df.ID_code, col_name] = sub_df['target'].values
    assert oof_feature[col_name].isnull().sum() == 0, 'invalid merger'
    return oof_feature


def mk_rank_oof_features(oof_filename, sub_filename, col_name):
    oof_feature = pd.DataFrame()
    oof_feature['ID_code'] = pd.read_pickle(
        './mnt/inputs/features/ID_code.pkl.gz').values
    oof_feature = oof_feature.set_index('ID_code')
    oof_df = pd.read_csv(oof_filename)
    sub_df = pd.read_csv(sub_filename)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=4221)
    # make it rank in each fold
    for trn_idx, val_idx in range(skf):
        oof_feature.iloc[val_idx] = (
            oof_feature.iloc[val_idx].rank() / len(val_idx)).values
    sub_df['target'] = (sub_df['target'].rank() / len(sub_df)).values
    assert (sub_df['target'].rank() / len(sub_df) ==
            sub_df['target'].values).mean() == 1, 'miss ranked features'
    oof_feature[col_name] = np.nan
    oof_feature.loc[oof_df.ID_code, col_name] = oof_df['oof_proba'].values
    oof_feature.loc[sub_df.ID_code, col_name] = sub_df['target'].values
    assert oof_feature[col_name].isnull().sum() == 0, 'invalid merger'
    return oof_feature
