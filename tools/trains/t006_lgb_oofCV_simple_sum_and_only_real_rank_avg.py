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
    # -- Prepare for training
    exp_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

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
    features = trn_tst_df.columns

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
    if 'nthread' not in PARAMS:
        PARAMS['nthread'] = os.cpu_count()

    sel_log('start training ...', None)
    oofs = []
    y_trues = []
    val_idxes = []
    scores = []
    best_iterations = []
    fold_importance_dict = {}
    cv_model = []
    for i, idxes in tqdm(list(enumerate(folds))):
        trn_idx, val_idx = idxes
        # -- Data resampling
        # Stock original data for validation
        fold_features_df, fold_target = value_resampling(
            features_df.iloc[trn_idx],
            target[trn_idx],
            configs['train']['sampling_type'],
            configs['train']['sampling_random_state'],
            configs['train']['os_lim'],
            configs['train']['pos_t'],
            configs['train']['neg_t'],
            logger=logger)

        # make lgbm dataset
        train_set = lightgbm.Dataset(fold_features_df,
                                     fold_target)
        valid_set = lightgbm.Dataset(features_df.iloc[val_idx],
                                     target[val_idx],)
        # train
        booster = lightgbm.train(
            params=PARAMS.copy(),
            train_set=train_set,
            num_boost_round=1000000,
            valid_sets=[valid_set, train_set],
            verbose_eval=1000,
            early_stopping_rounds=5000,
            callbacks=[log_evaluation(logger, period=1000)],
        )

        # predict using trained model
        y_pred = booster.predict(features_df.values[val_idx],
                                 num_iteration=None)
        y_true = target.values[val_idx]
        oofs.append(y_pred)
        y_trues.append(y_true)
        val_idxes.append(val_idx)

        # Calc AUC
        auc = roc_auc_score(y_true, y_pred)
        sel_log(f'fold AUC: {auc}', logger=logger)
        scores.append(auc)
        best_iterations.append(booster.best_iteration)

        # Save importance info
        fold_importance_df = pd.DataFrame()
        fold_importance_df['split'] = booster.feature_importance('split')
        fold_importance_df['gain'] = booster.feature_importance('gain')
        fold_importance_dict[i] = fold_importance_df

        # save model
        cv_model.append(booster)

    auc_mean, auc_std = np.mean(scores), np.std(scores)
    auc_oof = roc_auc_score(np.concatenate(y_trues), np.concatenate(oofs))
    best_iteration_mean = np.mean(best_iterations)
    sel_log(
        f'AUC_mean: {auc_mean:.5f}, AUC_std: {auc_std:.5f}',
        logger)
    sel_log(
        f'AUC OOF: {auc_oof}',
        logger)
    sel_log(
        f'BEST ITER MEAN: {best_iteration_mean}',
        logger)

    # -- Post processings
    filename_base = f'{script_name}_{exp_time}_{auc_mean:.5}'

    # Save oofs
    oof_df = pd.DataFrame()
    oof_df['ID_code'] = trn_ids.iloc[np.concatenate(val_idxes)]
    oof_df['y_val'] = np.concatenate(y_trues)
    oof_df['oof_proba'] = np.concatenate(oofs)
    oof_df = oof_df.set_index('ID_code').loc[trn_ids]
    oof_df.to_csv(
        './mnt/oofs/' +
        filename_base +
        '_oofs.csv',
        index=True)
    with open('./mnt/oofs/' + filename_base + '_oofs.pkl', 'wb') as fout:
        pickle.dump([val_idxes, oofs], fout)

    # Save importances
    # save_importance(configs['features'], fold_importance_dict,
    save_importance(features, fold_importance_dict,
                    './mnt/importances/' + filename_base + '_importances',
                    topk=100, main_metric='gain')

    # Save trained models
    with open('./mnt/trained_models/'
              + filename_base
              + '_models.pkl', 'wb') as fout:
        pickle.dump(cv_model, fout)

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
        submission_filename = f'./mnt/submissions/{filename_base}_sub.csv.gz'
        submission_filename_no_rank = f'./mnt/submissions/{filename_base}_sub_no_rank.csv'
        sel_log(f'saving submission file to {submission_filename}', logger)
        sub_df.to_csv(submission_filename, compression='gzip', index=False)
        sub_df_no_rank.to_csv(
            submission_filename_no_rank,
            index=False)
        if args.submit:
            os.system(
                f'kaggle competitions submit '
                f'santander-customer-transaction-prediction '
                f'-f {submission_filename} -m "{args.message}"')
    return auc_mean, auc_std, auc_oof
