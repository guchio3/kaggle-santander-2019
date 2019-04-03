import numpy as np
import pandas as pd
import scipy as sp
from tqdm import tqdm


def f031_graseck_normal_pdf_ratio_on_e036_features(df):
    res_df = pd.DataFrame()
    res_df['ID_code'] = df['ID_code']
    # do not use inplace for re-using
    df = df.set_index('ID_code')
    trn_id = pd.read_pickle('./mnt/inputs/nes_info/trn_ID_code.pkl.gz')
    tst_id = pd.read_pickle('./mnt/inputs/nes_info/tst_ID_code.pkl.gz')
    target = pd.read_pickle('./mnt/inputs/nes_info/target.pkl.gz')
    t0_rows = (target == 0).values
    t1_rows = (target == 1).values
    train_df = df.loc[trn_id]
    test_df = df.loc[tst_id]
    prefix = 'graseck_normal_pdf_ratio_'
    res_df = res_df.set_index('ID_code')
    # col rounding
    for col in tqdm(df.columns):
        res_df[prefix + col] = np.nan
        mean_t0 = train_df.loc[t0_rows, col].mean()
        mean_t1 = train_df.loc[t1_rows, col].mean()
        std_t0 = train_df.loc[t0_rows, col].std()
        std_t1 = train_df.loc[t1_rows, col].std()

        zval0 = (train_df[col].values - mean_t0) / std_t0
        zval1 = (train_df[col].values - mean_t1) / std_t1
        pval0 = (1 - sp.stats.norm.cdf(np.abs(zval0))) * 2
        pval1 = (1 - sp.stats.norm.cdf(np.abs(zval1))) * 2
        train_df[prefix + col] = np.log10(pval1) - np.log10(pval0)
        train_df[prefix + col] = pd.to_numeric(
                train_df[prefix + col], downcast='float')
        res_df.loc[trn_id, prefix + col] = train_df[prefix + col]

        zval0 = (test_df[col].values - mean_t0) / std_t0
        zval1 = (test_df[col].values - mean_t1) / std_t1
        pval0 = (1 - sp.stats.norm.cdf(np.abs(zval0))) * 2
        pval1 = (1 - sp.stats.norm.cdf(np.abs(zval1))) * 2
        test_df[prefix + col] = np.log10(pval1) - np.log10(pval0)
        test_df[prefix + col] = pd.to_numeric(
                test_df[prefix + col], downcast='float')
        res_df.loc[tst_id, prefix + col] = test_df[prefix + col]
    res_df = res_df.reset_index()

    # return as id is set to the index
    res_df = res_df.set_index('ID_code').add_prefix('f031_')
    return res_df
