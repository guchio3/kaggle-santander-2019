import numpy as np
import pandas as pd


def f003_900_kernel_round_features(df):
    res_df = pd.DataFrame()
    res_df['ID_code'] = df['ID_code']
    df.drop(['ID_code', 'target'], axis=1, inplace=True)
    # col rounding
    for col in df.columns:
        res_df[f'r1_{col}'] = np.round(df[col], 1)
        res_df[f'r2_{col}'] = np.round(df[col], 2)
    res_df = res_df.set_index('ID_code').add_prefix('f003_').reset_index()
    return res_df
