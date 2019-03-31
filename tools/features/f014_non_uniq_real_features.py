import numpy as np
import pandas as pd
from tqdm import tqdm


def f014_non_uniq_real_features(df):
    res_df = pd.DataFrame()
    res_df['ID_code'] = df['ID_code']
    # do not use inplace for re-using
    df = df.drop(['target'], axis=1)
    df = df.set_index('ID_code')
    trn_id = pd.read_pickle('./mnt/inputs/nes_info/trn_ID_code.pkl.gz')
    tst_id = pd.read_pickle('./mnt/inputs/nes_info/tst_ID_code.pkl.gz')
    reals = np.load('./mnt/inputs/nes_info/real_samples_indexes.npz.npy')
    # col rounding
    for col in tqdm(df.columns):
        trn_col = df.loc[trn_id][col]
        real_col = df.loc[tst_id][col].iloc[reals]
        uniq_cnt_dict = pd.concat(
            [trn_col, real_col], axis=0).value_counts().to_dict()
        res_df['non_uniq_real_' + col] = df[col].apply(
            lambda x: x if uniq_cnt_dict[x] != 1 else np.nan).values
    # return as id is set to the index
    res_df = res_df.set_index('ID_code').add_prefix('f014_')
    return res_df
