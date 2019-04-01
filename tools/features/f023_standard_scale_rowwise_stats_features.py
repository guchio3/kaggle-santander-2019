import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def f023_standard_scale_rowwise_stats_features(df):
    res_df = pd.DataFrame()
    res_df['ID_code'] = df['ID_code']
    # do not use inplace for re-using
    trn_id = pd.read_pickle('./mnt/inputs/nes_info/trn_ID_code.pkl.gz')
    tst_id = pd.read_pickle('./mnt/inputs/nes_info/tst_ID_code.pkl.gz')
    reals = np.load('./mnt/inputs/nes_info/real_samples_indexes.npz.npy')
    # standard scale
    df = df.set_index('ID_code')
    trn_df = df.loc[trn_id].reset_index(drop=True)
    tst_df = df.loc[tst_id]
    real_df = tst_df.iloc[reals.reset_index(drop=True)]
    fit_df = pd.concat([trn_df, real_df], axis=0)
    df = df.set_index('ID_code')
    ss = StandardScaler()
    ss.fit(fit_df)
    df = ss.transform(df)
    # horizontal stats
    res_df['ss_horizontal_sum'] = df.sum(axis=1)
    res_df['ss_horizontal_min'] = df.min(axis=1)
    res_df['ss_horizontal_max'] = df.max(axis=1)
    res_df['ss_horizontal_mean'] = df.mean(axis=1)
    res_df['ss_horizontal_std'] = df.std(axis=1)
    res_df['ss_horizontal_skew'] = df.skew(axis=1)
    res_df['ss_horizontal_kurt'] = df.kurtosis(axis=1)
    res_df['ss_horizontal_median'] = df.median(axis=1)
    # set prefix
    # return as id is set to the index
    res_df = res_df.set_index('ID_code').add_prefix('f023_')
    return res_df
