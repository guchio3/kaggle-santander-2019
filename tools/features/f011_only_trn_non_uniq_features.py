import numpy as np
import pandas as pd
from tqdm import tqdm


def f011_only_trn_non_uniq_features(df):
    res_df = pd.DataFrame()
    res_df['ID_code'] = df['ID_code']
    # do not use inplace for re-using
    df = df.drop(['target'], axis=1)
    df = df.set_index('ID_code')
    trn_id = pd.read_pickle('./mnt/inputs/nes_info/trn_ID_code.pkl.gz')
    # col rounding
    for col in tqdm(df.columns):
        uniq_cnt_dict = df.loc[trn_id][col].value_counts().to_dict()
        res_df['only_trn_non_uniq_masked_' + col] = df[col].apply(
            lambda x: x if x in uniq_cnt_dict
            and uniq_cnt_dict[x] != 1 else np.nan).values
    # return as id is set to the index
    res_df = res_df.set_index('ID_code').add_prefix('f011_')
    return res_df
