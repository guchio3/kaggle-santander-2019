import numpy as np
import pandas as pd
from tqdm import tqdm


def f017_filtered_uniq_cnt_encoding(df):
    res_df = pd.DataFrame()
    res_df['ID_code'] = df['ID_code']
    # do not use inplace for re-using
    df = df.drop(['ID_code'], axis=1)
    # col rounding
    for col in tqdm(df.columns):
        uniq_cnt_dict = df[col].value_counts().to_dict()
        res_df['filtered_uniq_cnt_' + col] = df[col].apply(
            lambda x: np.nan if np.isnan(x) else uniq_cnt_dict[x]).values
    # return as id is set to the index
    res_df = res_df.set_index('ID_code').add_prefix('f017_')
    return res_df
