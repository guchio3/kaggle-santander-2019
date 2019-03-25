def f001_raw_features(df):
    df.drop('target', axis=1, inplace=True)
    # return as id is set to the index
    df.set_index('ID_code', inplace=True)
    df = df.add_prefix('f001_')
    return df
