import numpy as np
import pandas as pd
from tqdm import tqdm


def f012_common_features(df):
    res_df = pd.DataFrame()
    res_df['ID_code'] = df['ID_code']
    # do not use inplace for re-using
    df = df.drop(['target'], axis=1)
    df = df.set_index('ID_code')
    trn_id = pd.read_pickle('./mnt/inputs/nes_info/trn_ID_code.pkl.gz')
    tst_id = pd.read_pickle('./mnt/inputs/nes_info/tst_ID_code.pkl.gz')
    # col rounding
    for col in tqdm(df.columns):
        trn_values = set(df.loc[trn_id][col].unique().tolist())
        tst_values = set(df.loc[tst_id][col].unique().tolist())
        valid_values = trn_values & tst_values
        res_df['trn_tst_common_' + col] = df[col].apply(
            lambda x: x if x in valid_values else np.nan).values
    # return as id is set to the index
    res_df = res_df.set_index('ID_code').add_prefix('f012_')
    return res_df
