import pandas as pd
from tqdm import tqdm


def f004_mode_shift_pow_features(df):
    res_df = pd.DataFrame()
    res_df['ID_code'] = df['ID_code']
    # do not use inplace for re-using
    df = df.drop(['ID_code', 'target'], axis=1)
    # col rounding
    for col in tqdm(df.columns.tolist()):
        res_df[f'mode_shift_pow_{col}'] = (
            df[col] - df[col].mode().values[0]) ** 2
    # return as id is set to the index
    res_df = res_df.set_index('ID_code').add_prefix('f004_')
    return res_df
