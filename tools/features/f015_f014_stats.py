import pandas as pd


def f015_f014_stats(df):
    res_df = pd.DataFrame()
    res_df['ID_code'] = df['ID_code']
    # do not use inplace for re-using
    df = df.drop(['ID_code'], axis=1)
    # horizontal stats
    res_df['horizontal_sum'] = df.sum(axis=1)
    res_df['horizontal_min'] = df.min(axis=1)
    res_df['horizontal_max'] = df.max(axis=1)
    res_df['horizontal_mean'] = df.mean(axis=1)
    res_df['horizontal_std'] = df.std(axis=1)
    res_df['horizontal_skew'] = df.skew(axis=1)
    res_df['horizontal_kurt'] = df.kurtosis(axis=1)
    res_df['horizontal_median'] = df.median(axis=1)
    # set prefix
    # return as id is set to the index
    res_df = res_df.set_index('ID_code').add_prefix('f015_')
    return res_df
