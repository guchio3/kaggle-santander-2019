import datetime
import os
import pickle
import warnings
from itertools import tee

import lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from tools.utils.features import (get_all_features, load_features,
                                  select_features)
from tools.utils.logs import dec_timer, log_evaluation, sel_log
from tools.utils.samplings import value_resampling
from tools.utils.visualizations import save_importance

warnings.simplefilter(action='ignore', category=FutureWarning)


NES_DIR = './mnt/inputs/nes_info/'
FEATURE_DIR = './mnt/inputs/features/'


@dec_timer
def t006_lgb_train(args, script_name, configs, logger):
    '''
    policy
    ------------
    * use original functions only if there's no pre-coded functions
        in useful libraries such as sklearn.

    todos
    ------------
    * load features
    * train the model
    * save the followings
        * logs
        * oofs
        * importances
        * trained models
        * submissions (if test mode)

    '''
    # -- Load train data
    sel_log('loading training data ...', None)
    trn_ids = pd.read_pickle(
        NES_DIR + 'trn_ID_code.pkl.gz', compression='gzip')
    tst_ids = pd.read_pickle(
        NES_DIR + 'tst_ID_code.pkl.gz', compression='gzip')
    target = pd.read_pickle(
        NES_DIR + 'target.pkl.gz', compression='gzip')
    if args.debug:
        sample_idxes = trn_ids.reset_index(
            drop=True).sample(
            random_state=71,
            frac=0.05).index
        target = target.iloc[sample_idxes].reset_index(drop=True)
        trn_ids = trn_ids.iloc[sample_idxes].reset_index(drop=True)

    # load features
    if configs['train']['all_features']:
        _features = get_all_features(FEATURE_DIR)
    else:
        _features = configs['features']
    trn_tst_df = load_features(_features, FEATURE_DIR, logger=logger)\
        .set_index('ID_code')

    # feature selection if needed
    if configs['train']['feature_selection']:
        trn_tst_df = select_features(trn_tst_df,
                                     configs['train']['feature_select_path'],
                                     configs['train']['metric'],
                                     configs['train']['feature_topk'])
    # split train and test
    sel_log(f'now splitting the df to train and test ones ...', None)
    features_df = trn_tst_df.loc[trn_ids].reset_index(drop=True)
    test_features_df = trn_tst_df.loc[tst_ids].reset_index(drop=True)

    # -- Split using group k-fold w/ shuffling
    # NOTE: this is not stratified, I wanna implement it in the future
    if configs['train']['fold_type'] == 'skf':
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=4221)
        folds = skf.split(features_df, target)
        configs['train']['single_model'] = False
    else:
        print(f"ERROR: wrong fold_type, {configs['train']['fold_type']}")
    folds, pred_folds = tee(folds)

    # -- Make training dataset
    # print shape
    sel_log(f'used features are {features_df.columns.tolist()}', logger)
    sel_log(f'the shape features_df is {features_df.shape}', logger)

    # -- CV
    # Set params
    PARAMS = configs['lgbm_params']
    PARAMS['nthread'] = os.cpu_count()

    sel_log('start training ...', None)

    with open('e071_e036_t006_w_pair_augmentation_7_7_2019-04-05-17-35-24_0.92357_models.pkl', 'rb') as fin:
        cv_model = pickle.load(fin)

    # --- Make submission file
    if args.test:
        if configs['train']['single_model']:
            # train single model
            best_iter = np.mean(
                [booster.best_iteration for booster in cv_model])
            single_train_set = lightgbm.Dataset(features_df, target.values)
            single_booster = lightgbm.train(
                params=PARAMS,
                num_boost_round=int(best_iter * 1.3),
                train_set=single_train_set,
                verbose_eval=1000,
                callbacks=[log_evaluation(logger, period=1000)],
            )
            # re-save model for prediction
            # cv_model.append(single_booster)

        # -- Prediction
        sel_log('predicting for test ...', None)
        preds = []
        preds_no_rank = []
        reals = np.load('./mnt/inputs/nes_info/real_samples_indexes.npz.npy')
        # for booster in tqdm(cv_model.boosters):
        for booster in tqdm(cv_model):
            pred = booster.predict(test_features_df.values, num_iteration=None)
            pred = pd.Series(pred)
            # rank avg only using real part
            preds_no_rank.append(pred.copy())
            pred.iloc[reals] = pred.iloc[reals].rank() / reals.shape
            preds.append(pred)
        if len(cv_model) > 1:
            target_values = np.mean(preds, axis=0)
            target_values_no_rank = np.mean(preds_no_rank, axis=0)
        else:
            target_values = preds[0]
            target_values_no_rank = preds[0]
        # blend single model
        if configs['train']['single_model']:
            pred = single_booster.predict(
                test_features_df.values, num_iteration=None)
            pred = pd.Series(pred)
            target_values = (target_values + (pred.rank() / pred.shape)) / 2

        # -- Make submission file
        sel_log(f'loading sample submission file ...', None)
        sub_df = pd.read_csv(
            './mnt/inputs/origin/sample_submission.csv.zip',
            compression='zip')
        sub_df.target = target_values
        sub_df_no_rank = pd.read_csv(
            './mnt/inputs/origin/sample_submission.csv.zip',
            compression='zip')
        sub_df_no_rank.target = target_values_no_rank

        # print stats
        submission_filename_no_rank = f'./mnt/submissions/temp.csv'
        sub_df_no_rank.to_csv(
            submission_filename_no_rank,
            index=False)
    return None, None, None
