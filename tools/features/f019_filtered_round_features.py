import numpy as np
import pandas as pd


def f019_filtered_round_features(df):
    res_df = pd.DataFrame()
    res_df['ID_code'] = df['ID_code']
    # do not use inplace for re-using
    df = df.drop(['ID_code'], axis=1)
    # col rounding
    for col in df.columns:
        res_df[f'r1_{col}'] = np.round(df[col], 1)
        res_df[f'r2_{col}'] = np.round(df[col], 2)
    # return as id is set to the index
    res_df = res_df.set_index('ID_code').add_prefix('f019_')
    return res_df
