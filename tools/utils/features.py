import gc
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .logs import dec_timer, sel_log


@dec_timer
def split_df(base_df, target_df, split_name,
             target_name, n_sections, logger=None):
    '''
    policy
    ------------
    * split df based on split_id, and set split_id as index
        because of efficiency.

    '''
    sel_log(
        f'now splitting a df to {n_sections} dfs using {split_name} ...',
        logger)
    split_ids = base_df[split_name].unique()
    splitted_ids = np.array_split(split_ids, n_sections)
    if split_name == target_name:
        target_ids = splitted_ids
    else:
        target_ids = [base_df.set_index(split_name)
                      .loc[splitted_id][target_name]
                      for splitted_id in splitted_ids]
    # Pay attention that this is col-wise splitting bacause of the
    #   data structure of this competition.
    target_df = target_df.set_index(target_name)
    dfs = [target_df.loc[target_id.astype(str)].reset_index()
           for target_id in target_ids]
    return dfs


def get_all_features(path):
    files = os.listdir(path)
    features = [_file.split('.')[0] for _file in files]
    return features


def _load_feature(feature, base_dir, logger=None):
    load_filename = base_dir + feature + '.pkl.gz'
    # sel_log(f'loading from {load_filename} ...', logger)
    feature = pd.read_pickle(load_filename, compression='gzip')
    # drop index because its very heavy to concat, and already sorted.
    feature.reset_index(drop=True, inplace=True)
    return feature


@dec_timer
def load_features(features, base_dir, nthread=os.cpu_count(), logger=None):
    loaded_features = []
    sel_log(f'now loading features ... ', None)
    with Pool(nthread) as p:
        iter_func = partial(_load_feature, base_dir=base_dir, logger=logger)
        loaded_features = p.map(iter_func, features)
        p.close()
        p.join()
        gc.collect()
    sel_log(f'now concatenating the loaded features ... ', None)
    features_df = pd.concat(loaded_features, axis=1)[features]
    return features_df


def _save_feature(feature_pair, base_dir, logger=None):
    feature, feature_df = feature_pair
    save_filename = base_dir + feature + '.pkl.gz'
    if os.path.exists(save_filename):
        sel_log(f'already exists at {save_filename} !', None)
    else:
        sel_log(f'saving to {save_filename} ...', logger)
        feature_df.reset_index(drop=True, inplace=True)
        feature_df.to_pickle(save_filename, compression='gzip')


@dec_timer
def save_features(features_df, base_dir, nthread=os.cpu_count(), logger=None):
    feature_pairs = [[feature, features_df[feature]] for feature in
                     features_df.columns]
    with Pool(nthread) as p:
        iter_func = partial(_save_feature, base_dir=base_dir, logger=logger)
        _ = p.map(iter_func, feature_pairs)
        p.close()
        p.join()
        del _


def select_features(df, importance_csv_path, metric='gain_mean', topk=10):
    #    if metric == :
    importance_df = pd.read_csv(importance_csv_path)
    importance_df.sort_values(metric, ascending=False, inplace=True)
    selected_df = df[importance_df.head(topk).features]
    return selected_df


@dec_timer
def _mk_features(load_func, feature_func, feature_ids, nthread=os.cpu_count(),
                 trn_tst_df=None, trn_df=None, tst_df=None,
                 logger=None):
    # Load dfs
    # Does not load if the feature_ids are not the targets.
    trn_tst_df, trn_df, tst_df = load_func(
        feature_ids, trn_tst_df, trn_df, tst_df, logger)
    # Finish before feature engineering if the feature_ids are not the targets.
    if trn_tst_df is None:
        return None, None, None

    # split df using ids to speed up feature engineering
    series_dfs = split_df(
        trn_tst_df,
        trn_tst_df,
        'ID_code',
        'ID_code',
        # trn_tst_df['ID_code'].nunique(),
        nthread,
        logger=logger)

    with Pool(nthread) as p:
        sel_log(f'feature engineering ...', None)
        # Using partial enable to use constant argument for the iteration.
        iter_func = partial(feature_func, feature_ids=feature_ids)
        features_list = p.map(iter_func, series_dfs)
        p.close()
        p.join()
        features_df = pd.concat(features_list, axis=0)

    # Merge w/ meta.
    # This time, i don't remove the original features because
    #   this is the base feature function.
    ids = pd.DataFrame(trn_tst_df.ID_code)
    sel_log(f'merging features ...', None)
    trn_tst_df = ids.merge(
        features_df, on='ID_code', how='left')

    # Save the features
    feature_dir = './mnt/inputs/features/'
    sel_log(f'saving features ...', logger)
    save_features(trn_tst_df, feature_dir, nthread, logger)

    return trn_tst_df, trn_df, tst_df
