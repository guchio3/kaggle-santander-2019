import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


def f032_e036_ss_pow_features(df):
    res_df = pd.DataFrame()
    res_df['ID_code'] = df['ID_code']
    # do not use inplace for re-using
    trn_id = pd.read_pickle('./mnt/inputs/nes_info/trn_ID_code.pkl.gz')
    tst_id = pd.read_pickle('./mnt/inputs/nes_info/tst_ID_code.pkl.gz')
    reals = np.load('./mnt/inputs/nes_info/real_samples_indexes.npz.npy')
    # standard scale
    df = df.set_index('ID_code')
    trn_df = df.loc[trn_id].reset_index(drop=True)
    tst_df = df.loc[tst_id]
    real_df = tst_df.iloc[reals].reset_index(drop=True)
    fit_df = pd.concat([trn_df, real_df], axis=0)
    ss = StandardScaler()
    ss.fit(fit_df)
    df = pd.DataFrame(ss.transform(df))
    # col rounding
    for col in tqdm(df.columns.tolist()):
        res_df[f'ss_pow_{col}'] = (df[col] ** 2).values
    # return as id is set to the index
    res_df = res_df.set_index('ID_code').add_prefix('f032_')
    return res_df
