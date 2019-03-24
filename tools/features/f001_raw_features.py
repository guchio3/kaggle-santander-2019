def f001_raw_features(df):
    df.drop('target', axis=1, inplace=True)
    df.set_index('ID_code', inplace=True)
    df = df.add_prefix('f001_')
    df.reset_index(inplace=True)
    return df
