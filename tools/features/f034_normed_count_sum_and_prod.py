import numpy as np
import pandas as pd
from tqdm import tqdm


def f034_normed_count_sum_and_prod(df):
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
        res_df[col] = df[col].apply(
            lambda x: np.nan if np.isnan(x) else uniq_cnt_dict[x]).values
    res_df = res_df.set_index('ID_code')
    normed_uniq_cnt_sum = ((res_df == 1) / res_df.nunique()).sum(axis=1)
    normed_uniq_cnt_prod = ((res_df == 1) / res_df.nunique()).prod(axis=1)
    normed_non_uniq_cnt_sum = ((res_df != 1) / res_df.nunique()).sum(axis=1)
    normed_non_uniq_cnt_prod = ((res_df != 1) / res_df.nunique()).prod(axis=1)
    for col in tqdm(res_df.columns):
        res_df[col] = res_df[col] / df[col].nunique()
    normed_cnt_sum = res_df.sum(axis=1)
    normed_cnt_prod = res_df.prod(axis=1)
    res_df = res_df.reset_index()
    res_df['normed_cnt_sum'] = normed_cnt_sum
    res_df['normed_cnt_prod'] = normed_cnt_prod
    res_df['normed_uniq_cnt_sum'] = normed_uniq_cnt_sum
    res_df['normed_non_uniq_cnt_sum'] = normed_non_uniq_cnt_sum
    res_df['normed_uniq_cnt_prod'] = normed_uniq_cnt_prod
    res_df['normed_non_uniq_cnt_prod'] = normed_non_uniq_cnt_prod
    res_df = res_df[
            [
                'ID_code',
                'normed_cnt_sum',
                'normed_cnt_prod',
                'normed_uniq_cnt_sum',
                'normed_non_uniq_cnt_sum',
                'normed_uniq_cnt_prod',
                'normed_non_uniq_cnt_prod',
            ]
        ]
    # return as id is set to the index
    res_df = res_df.set_index('ID_code').add_prefix('f034_')
    return res_df
