import numpy as np
import pandas as pd
from tqdm import tqdm


def f010_non_uniq_features(df):
    res_df = pd.DataFrame()
    res_df['ID_code'] = df['ID_code']
    # do not use inplace for re-using
    df = df.drop(['ID_code', 'target'], axis=1)
    # col rounding
    for col in tqdm(df.columns):
        uniq_cnt_dict = df[col].value_counts().to_dict()
        res_df['non_uniq_masked_' + col] = df[col].apply(
            lambda x: x if uniq_cnt_dict[x] != 1 else np.nan)
    # return as id is set to the index
    res_df = res_df.set_index('ID_code').add_prefix('f010_')
    return res_df
