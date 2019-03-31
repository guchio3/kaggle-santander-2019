import pandas as pd


def f016_nan_counts(df):
    res_df = pd.DataFrame()
    res_df['ID_code'] = df['ID_code']
    # do not use inplace for re-using
    df = df.drop(['ID_code'], axis=1)
    # horizontal stats
    res_df['nan_cnt'] = df.isnull().sum(axis=1)
    # set prefix
    # return as id is set to the index
    res_df = res_df.set_index('ID_code').add_prefix('f016_')
    return res_df

