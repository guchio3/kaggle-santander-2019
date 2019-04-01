import numpy as np
import pandas as pd
from tqdm import tqdm

from .logs import sel_log


class TargetEncoder():
    def __init__(self, logger=None):
        self.logger = logger
        self.encoding_dict = {}

    def fit(self, features_df, target):
        if len(self.encoding_dict):
            sel_log('Warning, the encoding dict already exist.', self.logger)
        sel_log('now fitting for target encoding ...', self.logger)
        target_cols = features_df.columns
        features_df = pd.concat([features_df, target], axis=1)
        for col in tqdm(target_cols):
            if col in self.encoding_dict:
                sel_log(f'WARNING, the col {col} is already fit.'
                        ' Now re-fitting ...', self.logger)
            _col_dict = features_df[[col, 'target']].groupby(col).target.\
                mean().to_dict()
            self.encoding_dict[col] = _col_dict

    def transform(self, base_df):
        sel_log('now transforming for target encoding ...', self.logger)
        res_df = base_df.copy()
        for col in tqdm(base_df.columns):
            res_df[col] = base_df[col].\
                apply(lambda x: self.encoding_dict[col][x]
                      if x in self.encoding_dict[col] else np.nan).values
        res_df = res_df.add_prefix('te_')
        return res_df


# olivier method
# https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None,
                  tst_series=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[
        target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / \
        (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * \
        (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(
            columns={
                'index': target.name,
                target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(
            columns={
                'index': target.name,
                target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)
