import pandas as pd
from tqdm import tqdm


def f007_uniq_cnt_meta_features(df):
    res_df = pd.DataFrame()
    res_df['ID_code'] = df['ID_code']
    # do not use inplace for re-using
    df = df.drop(['ID_code'], axis=1)
    # mk features
    res_df['uniq_cnt_sum'] = df.sum(axis=1)
    res_df['onlyone_cnt'] = (df == 1).sum(axis=1)
    # return as id is set to the index
    res_df = res_df.set_index('ID_code').add_prefix('f007_')
    return res_df
