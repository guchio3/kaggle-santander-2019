import numpy as np
import pandas as pd


def f002_900_kernel_features(df):
    res_df = pd.DataFrame()
    res_df['ID_code'] = df['ID_code']
    df.drop(['ID_code', 'target'], axis=1, inplace=True)
    # horizontal stats
    res_df['horizontal_sum'] = df.sum(axis=1)
    res_df['horizontal_min'] = df.min(axis=1)
    res_df['horizontal_max'] = df.max(axis=1)
    res_df['horizontal_mean'] = df.mean(axis=1)
    res_df['horizontal_std'] = df.std(axis=1)
    res_df['horizontal_skew'] = df.skew(axis=1)
    res_df['horizontal_kurt'] = df.kurtosis(axis=1)
    res_df['horizontal_median'] = df.median(axis=1)
    # col rounding
    for col in df.columns:
        res_df[f'r1_{col}'] = np.round(df[col], 1)
        res_df[f'r2_{col}'] = np.round(df[col], 2)
    res_df = res_df.set_index('ID_code').add_prefix('f002_').reset_index()
    return res_df
