import pandas as pd
from tqdm import tqdm


def f005_raw_sum_features(df):
    res_df = pd.DataFrame()
    res_df['ID_code'] = df['ID_code']
    # do not use inplace for re-using
    df = df.drop(['ID_code', 'target'], axis=1)
    # col rounding
    # return as id is set to the index
    res_df = res_df.set_index('ID_code').add_prefix('f005_')
    return res_df
