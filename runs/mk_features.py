import os
import sys
import time
from logging import getLogger

from tools.features.u001_base_feature_utils import (_base_features,
                                                    _load_base_features)
from tools.features.u002_colwise_features import (_colwise_features,
                                                  _load_colwise_features)
from tools.features.u003_colwise_meta_features import (_colwise_meta_features,
                                                       _load_colwise_meta_features)
from tools.features.u004_meta_features import (_load_meta_features,
                                               _meta_features)
from tools.utils.args import parse_feature_args
from tools.utils.features import _mk_colwise_features, _mk_features
from tools.utils.logs import dec_timer, logInit, send_line_notification


@dec_timer
def mk_features(args, logger):
    trn_df = None
    tst_df = None
    trn_tst_df = None
    # base features
    trn_tst_df, trn_meta_df, tst_meta_df = _mk_features(
        _load_base_features, _base_features,
        args.feature_ids, os.cpu_count(), trn_tst_df,
        trn_df, tst_df, logger=logger)
    trn_tst_df, trn_meta_df, tst_meta_df = _mk_colwise_features(
        _load_colwise_features, _colwise_features,
        args.feature_ids, os.cpu_count(), trn_tst_df,
        trn_df, tst_df, logger=logger)
    trn_tst_df, trn_meta_df, tst_meta_df = _mk_colwise_features(
        _load_colwise_meta_features, _colwise_meta_features,
        args.feature_ids, os.cpu_count(), trn_tst_df,
        trn_df, tst_df, logger=logger)
    trn_tst_df, trn_meta_df, tst_meta_df = _mk_features(
        _load_meta_features, _meta_features,
        args.feature_ids, os.cpu_count(), trn_tst_df,
        trn_df, tst_df, logger=logger)


if __name__ == '__main__':
    t0 = time.time()
    logger = getLogger(__name__)
    logger = logInit(logger, './mnt/logs/', 'mk_features.log')
    args = parse_feature_args(logger)

    logger.info('')
    logger.info('')
    logger.info(
        f'============ EXP {args.feature_ids}, '
        f'START MAKING FEATURES =============')
    mk_features(args, logger)
    prec_time = time.time() - t0
    send_line_notification(
        f'Finished: {" ".join(sys.argv)} in {prec_time:.1f} s !')
