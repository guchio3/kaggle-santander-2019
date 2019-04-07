import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold


def f035_trn_tst_common_target_encoding_features(df):
    res_df = pd.DataFrame()
    res_df['ID_code'] = df['ID_code']
    # do not use inplace for re-using
    df = df.drop(['target'], axis=1)
    df = df.set_index('ID_code')
    trn_id = pd.read_pickle('./mnt/inputs/nes_info/trn_ID_code.pkl.gz')
    tst_id = pd.read_pickle('./mnt/inputs/nes_info/tst_ID_code.pkl.gz')
    reals = np.load('./mnt/inputs/nes_info/real_samples_indexes.npz.npy')
    target = pd.read_pickle('./mnt/inputs/nes_info/target.pkl.gz')
    # col rounding
    for col in tqdm(df.columns):
        trn_col = df.loc[trn_id][col]
        real_col = df.loc[tst_id][col].iloc[reals]
        trn_values = set(trn_col.unique().tolist())
        real_values = set(real_col.unique().tolist())
        valid_values = trn_values & real_values
        res_df['trn_tst_non_common_' + col] = df[col].apply(
            lambda x: x if x in valid_values else np.nan).values
    # make fit_df
    res_df = res_df.set_index('ID_code')
    fit_df = res_df.loc[trn_id].copy()
    # fold-wise target encoding
    for col in tqdm(res_df.columns):
        # make fold
        skf = StratifiedKFold(10, random_state=71)
        folds = skf.split(fit_df[col], target)

        trn_col = df.loc[trn_id][col]
        real_col = df.loc[tst_id][col].iloc[reals]
        uniq_cnt_dict = pd.concat(
            [trn_col, real_col], axis=0).value_counts().to_dict()
        res_df['real_count_encoding_' + col] = df[col].apply(
            lambda x: np.nan if np.isnan(x) else uniq_cnt_dict[x]).values
    # return as id is set to the index
    res_df = res_df.set_index('ID_code').add_prefix('f035_')
    return res_df
