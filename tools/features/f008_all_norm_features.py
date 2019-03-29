import numpy as np
import pandas as pd


def f008_all_norm_features(df):
    res_df = pd.DataFrame()
    res_df['ID_code'] = df['ID_code']
    # do not use inplace for re-using
    df = df.drop(['ID_code', 'target'], axis=1)
    # mk_features
    res_df['all_norm'] = df.apply(
        lambda row: np.linalg.norm(row[2:]), axis=1)
    # return as id is set to the index
    res_df = res_df.set_index('ID_code').add_prefix('f008_')
    return res_df
