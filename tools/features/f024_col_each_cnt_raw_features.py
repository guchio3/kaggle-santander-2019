import numpy as np
import pandas as pd
from tqdm import tqdm


def f024_col_each_cnt_raw_features(df):
    res_df = pd.DataFrame()
    res_df['ID_code'] = df['ID_code']
    # do not use inplace for re-using
    df = df.drop(['target'], axis=1)
    df = df.set_index('ID_code')
    trn_id = pd.read_pickle('./mnt/inputs/nes_info/trn_ID_code.pkl.gz')
    tst_id = pd.read_pickle('./mnt/inputs/nes_info/tst_ID_code.pkl.gz')
    reals = np.load('./mnt/inputs/nes_info/real_samples_indexes.npz.npy')
    # get col each cnt raw
    for target_cnt in tqdm(range(1, 30)):
        for col in tqdm(df.columns):
            trn_col = df.loc[trn_id][col]
            real_col = df.loc[tst_id][col].iloc[reals]
            uniq_cnt_dict = pd.concat(
                [trn_col, real_col], axis=0).value_counts().to_dict()
            res_df['non_uniq_real_' + col] = df[col]\
                .apply(lambda x: x if uniq_cnt_dict[x]
                       != target_cnt else np.nan).values
    # set prefix
    # return as id is set to the index
    res_df = res_df.set_index('ID_code').add_prefix('f024_')
    return res_df
