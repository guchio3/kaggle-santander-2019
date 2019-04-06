import numpy as np
import pandas as pd
from tqdm import tqdm


def f033_real_count_encoding(df):
    res_df = pd.DataFrame()
    res_df['ID_code'] = df['ID_code']
    # do not use inplace for re-using
    df = df.set_index(['ID_code'])
    trn_id = pd.read_pickle('./mnt/inputs/nes_info/trn_ID_code.pkl.gz')
    tst_id = pd.read_pickle('./mnt/inputs/nes_info/tst_ID_code.pkl.gz')
    reals = np.load('./mnt/inputs/nes_info/real_samples_indexes.npz.npy')
    # col rounding
    for col in tqdm(df.columns):
        trn_col = df.loc[trn_id][col]
        real_col = df.loc[tst_id][col].iloc[reals]
        uniq_cnt_dict = pd.concat(
            [trn_col, real_col], axis=0).value_counts().to_dict()
        res_df['real_count_encoding_' + col] = df[col].apply(
            lambda x: np.nan if np.isnan(x) else uniq_cnt_dict[x]).values
#        assert res_df.loc[trn_id].isnull().sum() == 0, 'there exist nan in real'
#        assert res_df.loc[tst_id].iloc[reals].isnull().sum() == 0,\
#        'there exist nan in real'
    # return as id is set to the index
    res_df = res_df.set_index('ID_code').add_prefix('f033_')
    return res_df
