import datetime
import os
import pickle
import sys
import time
import warnings
from itertools import tee
from logging import getLogger

import lightgbm
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz, vstack
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from tools.utils.feature_tools import get_all_features, load_features
from tools.utils.foldings import TimeSplitSingleFold
from tools.utils.general_utils import (dec_timer, get_locs, load_configs,
                                       log_evaluation, logInit, parse_args,
                                       sel_log, send_line_notification,
                                       test_commit)
from tools.utils.metrics import eval_auc
from tools.utils.visualizations import save_importance

warnings.simplefilter(action='ignore', category=FutureWarning)


@dec_timer
def train(args, logger):
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
    nes_dir = './mnt/inputs/nes_info/'
    feature_dir = './mnt/inputs/features/'
    configs = load_configs('./configs/config.yml', logger)

    # -- Load train data
    sel_log('loading training data ...', None)
    trn_ids = pd.read_pickle(
        nes_dir + 'trn_ID_code.pkl.gz', compression='gzip')
    tst_ids = pd.read_pickle(
        nes_dir + 'tst_ID_code.pkl.gz', compression='gzip')
    target = pd.read_pickle(
        nes_dir + 'HasDetections.pkl.gz', compression='gzip')
    val_flgs = pd.read_pickle(nes_dir + 'val_flg.pkl')
    if configs['train']['debug']:
        sample_idxes = trn_ids.reset_index(
            drop=True).sample(
            random_state=71,
            frac=0.05).index
        target = target.iloc[sample_idxes].reset_index(drop=True)
        trn_ids = trn_ids.iloc[sample_idxes].reset_index(drop=True)
        val_flgs = val_flgs.iloc[sample_idxes].reset_index(drop=True)

    if configs['train']['all_features']:
        _features = get_all_features('./inputs/features/')
    else:
        _features = configs['features']
    trn_tst_df = load_features(_features, feature_dir, args.nthread, logger)\
        .set_index('MachineIdentifier')

    # feature selection if needed
#    if configs['train']['feature_selection']:
#        features_df = select_features(features_df,
#                                      configs['train']['feature_select_path'],
#                                      configs['train']['metric'],
#                                      configs['train']['feature_topk'])
#        test_features_df = select_features(test_features_df,
#                                           configs['train']['feature_select_path'],
#                                           configs['train']['metric'],
#                                           configs['train']['feature_topk'])

    features = trn_tst_df.columns
    # clarify the used categorical features
    # also encoding categorical features
    if configs['categorical_features']:
        categorical_features = sorted(
            list(set(configs['categorical_features']) & set(features)))
    else:
        categorical_features = []

    # split train and test
    sel_log(f'now splitting the df to train and test ones ...', None)
    features_df = trn_tst_df.loc[trn_ids].reset_index(drop=True)
    test_features_df = trn_tst_df.loc[tst_ids].reset_index(drop=True)

    # remove invalid features
###    features_df.drop(configs['invalid_features'], axis=1, inplace=True)

    # categorical_features = get_locs(
    #     features_df, configs['categorical_features'])

    # -- Split using group k-fold w/ shuffling
    # NOTE: this is not stratified, I wanna implement it in the future
    if configs['train']['fold_type'] == 'skf':
        skf = StratifiedKFold(configs['train']['fold_num'], random_state=71)
        folds = skf.split(features_df, target)
        configs['train']['single_model'] = False
        if 'e005_avsig_ts_diff' in features_df.columns:
            features_df['e005_avsig_ts_diff'] = pd.read_pickle(
                'inputs/nes_info/e005_avsig_ts_diff_full_trn.pkl.gz', compression='gzip')
        if 'e005_osver_ts_diff' in features_df.columns:
            features_df['e005_osver_ts_diff'] = pd.read_pickle(
                'inputs/nes_info/e005_osver_ts_diff_full_trn.pkl.gz', compression='gzip')
        if 'e008_datebl_ts_diff' in features_df.columns:
            features_df['e008_datebl_ts_diff'] = pd.read_pickle(
                'inputs/nes_info/e008_datebl_ts_diff_full_trn.pkl.gz', compression='gzip')
    elif configs['train']['fold_type'] == 'tssf':
        tssf = TimeSplitSingleFold()
        folds = tssf.split(val_flgs)
        configs['train']['single_model'] = True
    else:
        print(f"ERROR: wrong fold_type, {configs['train']['fold_type']}")
    folds, pred_folds = tee(folds)

    # -- Make training dataset
#    train_set = mlgb.Dataset(features_df, target,
#                             categorical_feature=categorical_features)
#    train_set = mlgb.Dataset(features_df.values, target.values,)
#                             feature_name=features,
#                             categorical_feature=configs['categorical_features'])

    # ohe
    if configs['train']['ohe']:
        sel_log('one hot encoding!', None)
        # drop categories
        # for col in categorical_features:
        for col in features_df.columns:
            features_df.loc[features_df[col].isnull().values
                            | (features_df[col] == np.inf).values,
                            col] = features_df[col].mode().values
            features_df[col] = features_df[col].astype(int)
        cat_place = [
            i for i in range(features_df.shape[1])
            if features_df.columns[i] in categorical_features]
        # fit one hot encoder
        ohe = OneHotEncoder(
            categorical_features=cat_place,
            sparse=False,
            dtype='uint8')
        ohe.fit(features_df)
        features_df = pd.DataFrame(ohe.transform(features_df))
#        features_cat_df = vstack([ohe.transform(features_cat_df[i * m:(i + 1) * m])
# for i in range(features_cat_df.shape[0] // m + 1)])
        # only used for test
        if args.test:
            sel_log('test one hot encoding!', None)
            for col in test_features_df.columns:
                test_features_df.loc[test_features_df[col].isnull().values
                                     | (test_features_df[col] == np.inf).values,
                                     col] = test_features_df[col].mode().values
                test_features_df[col] = test_features_df[col].astype(int)
            test_features_df = pd.DataFrame(ohe.transform(test_features_df))
        categorical_features = 'auto'

    # print shape
    sel_log(f'the shape features_df is {features_df.shape}', logger)

    # -- CV
    # Set params
    PARAMS = configs['lgbm_params']
    PARAMS['nthread'] = args.nthread
    # PARAMS['categorical_feature'] = categorical_features

    sel_log('start training ...', None)
    cv_model = []
    for i, idxes in tqdm(list(enumerate(folds))):
        trn_idx, val_idx = idxes
        # -- Data resampling
        # Stock original data for validation
#        if configs['preprocess']['resampling']:
#            trn_idx = resampling(outliers[trn_idx],
#                                 configs['preprocess']['resampling_type'],
#                                 configs['preprocess']['resampling_seed'],
#                                 configs['preprocess']['os_lim'])
        train_set = lightgbm.Dataset(features_df.iloc[trn_idx],
                                     target[trn_idx],)
#                                     categorical_feature=categorical_features)
        valid_set = lightgbm.Dataset(features_df.iloc[val_idx],
                                     target[val_idx],)
#                                     categorical_feature=categorical_features)

        booster = lightgbm.train(
            params=PARAMS.copy(),
            train_set=train_set,
            # feval=eval_auc,
            num_boost_round=20000,
            valid_sets=[valid_set, train_set],
            verbose_eval=100,
            early_stopping_rounds=200,
#            categorical_feature=categorical_features,
            callbacks=[log_evaluation(logger, period=100)],
        )
        cv_model.append(booster)
#    hist, cv_model = mlgb.cv(
#        params=PARAMS,
#        num_boost_round=10000,
#        folds=folds,
#        train_set=train_set,
#        verbose_eval=100,
#        early_stopping_rounds=200,
#        metrics='rmse',
#        callbacks=[log_evaluation(logger, period=100)],
#    )

    # -- Prediction
    if configs['train']['single_model']:
        # reuse val_idx here
        booster = cv_model[0]
        y_pred = booster.predict(features_df.values[val_idx],
                                 num_iteration=None)
        y_true = target.values[val_idx]
        oofs = [y_pred]
        y_trues = [y_true]
        val_idxes = [val_idx]
        auc = roc_auc_score(y_true, y_pred)
        scores = [auc]

        fold_importance_df = pd.DataFrame()
        fold_importance_df['split'] = booster.feature_importance('split')
        fold_importance_df['gain'] = booster.feature_importance('gain')
        fold_importance_dict = {0: fold_importance_df}
    else:
        sel_log('predicting using cv models ...', logger)
        oofs = []
        y_trues = []
        val_idxes = []
        scores = []
        fold_importance_dict = {}
        for i, idxes in tqdm(list(enumerate(pred_folds))):
            trn_idx, val_idx = idxes
            # booster = cv_model.boosters[i]
            booster = cv_model[i]

            # Get and store oof and y_true
            y_pred = booster.predict(features_df.values[val_idx],
                                     num_iteration=None)
            y_true = target.values[val_idx]
            oofs.append(y_pred)
            y_trues.append(y_true)
            val_idxes.append(val_idx)

            # Calc AUC
            auc = roc_auc_score(y_true, y_pred)
            scores.append(auc)

            # Save importance info
            fold_importance_df = pd.DataFrame()
            fold_importance_df['split'] = booster.feature_importance('split')
            fold_importance_df['gain'] = booster.feature_importance('gain')
            fold_importance_dict[i] = fold_importance_df

    auc_mean, auc_std = np.mean(scores), np.std(scores)
    sel_log(
        f'AUC_mean: {auc_mean:.4f}, AUC_std: {auc_std:.4f}',
        logger)

    # -- Post processings
    filename_base = f'{args.exp_ids[0]}_{exp_time}_{auc_mean:.4}'

    # Save oofs
    with open('./oofs/' + filename_base + '_oofs.pkl', 'wb') as fout:
        pickle.dump([val_idxes, oofs], fout)

    # Save importances
    # save_importance(configs['features'], fold_importance_dict,
    if not configs['train']['ohe']:
        save_importance(features, fold_importance_dict,
                        './importances/' + filename_base + '_importances',
                        topk=100, main_metric='split')

    # Save trained models
    with open(
            './trained_models/' + filename_base + '_models.pkl', 'wb') as fout:
        pickle.dump(cv_model, fout)

    # --- Make submission file
    if args.test:
        if configs['train']['single_model']:
            # reload features of timediff from the last day of each dataset
            if 'e005_avsig_ts_diff' in features_df.columns:
                features_df['e005_avsig_ts_diff'] = pd.read_pickle(
                    'inputs/nes_info/e005_avsig_ts_diff_full_trn.pkl.gz', compression='gzip')
            if 'e005_osver_ts_diff' in features_df.columns:
                features_df['e005_osver_ts_diff'] = pd.read_pickle(
                    'inputs/nes_info/e005_osver_ts_diff_full_trn.pkl.gz', compression='gzip')
            if 'e008_datebl_ts_diff' in features_df.columns:
                features_df['e008_datebl_ts_diff'] = pd.read_pickle(
                    'inputs/nes_info/e008_datebl_ts_diff_full_trn.pkl.gz', compression='gzip')
            # train single model
            best_iter = cv_model[0].best_iteration
            single_train_set = lightgbm.Dataset(features_df, target.values)
            single_booster = lightgbm.train(
                params=PARAMS,
                num_boost_round=int(best_iter * 1.4),
                train_set=single_train_set,
                # valid_sets=[single_train_set],
                verbose_eval=100,
                # early_stopping_rounds=200,
                # categorical_feature=categorical_features,
                callbacks=[log_evaluation(logger, period=100)],
            )
            # re-save model for prediction
            cv_model = [single_booster]

        #        # -- Prepare for test
        #        test_base_dir = './inputs/test/'
        #
        #        sel_log('loading test data ...', None)
        #        test_features_df = load_features(
        #            features, test_base_dir, logger)
        #        # label encoding
        #        sel_log('encoding categorical features ...', None)
        #        test_features_df = fill_unseens(features_df, test_features_df,
        #                                        configs['categorical_features'],
        #                                        args.nthread)
        #        test_features_df, le_dict = label_encoding(test_features_df, le_dict)

        # -- Prediction
        sel_log('predicting for test ...', None)
        preds = []
        # for booster in tqdm(cv_model.boosters):
        for booster in tqdm(cv_model):
            pred = booster.predict(test_features_df.values, num_iteration=None)
            pred = pd.Series(pred)
            preds.append(pred.rank() / pred.shape)
        if len(cv_model) > 1:
            target_values = np.mean(preds, axis=0)
        else:
            target_values = preds[0]

        # -- Make submission file
        sel_log(f'loading sample submission file ...', None)
        sub_df = pd.read_csv(
            './inputs/origin/sample_submission.csv.zip',
            compression='zip')
        sub_df.HasDetections = target_values

        # print stats
        submission_filename = f'./submissions/{filename_base}_sub.csv.gz'
        sel_log(f'saving submission file to {submission_filename}', logger)
        sub_df.to_csv(submission_filename, compression='gzip', index=False)
        if args.submit:
            os.system(
                f'kaggle competitions submit microsoft-malware-prediction -f {submission_filename} -m "{args.message}"')


if __name__ == '__main__':
    t0 = time.time()
    logger = getLogger(__name__)
    logger = logInit(logger, './logs/', 'lgb_train.log')
    args = parse_args(logger)

    logger.info('')
    logger.info('')
    logger.info(
        f'============ EXP {args.exp_ids[0]}, START TRAINING =============')
    train(args, logger)
    test_commit(args, './logs/test_commit.log')
    prec_time = time.time() - t0
    send_line_notification(
        f'Finished: {" ".join(sys.argv)} in {prec_time:.1f} s !')
