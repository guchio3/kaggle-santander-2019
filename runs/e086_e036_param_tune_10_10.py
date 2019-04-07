import os
import time
from logging import getLogger

from tools.trains.t006_lgb_oofCV_simple_sum_and_only_real_rank_avg import \
    t006_lgb_train as train
from tools.utils.args import parse_train_args
from tools.utils.configs import load_configs
from tools.utils.logs import logInit, send_line_notification

CONFIG_FILE = './configs/c086.yml'

if __name__ == '__main__':
    t0 = time.time()
    script_name = os.path.basename(__file__).split('.')[0]
    log_file = script_name + '.log'

    logger = getLogger(__name__)
    logger = logInit(logger, './mnt/logs/', log_file)
    args = parse_train_args(logger)
    configs = load_configs(CONFIG_FILE, logger)

    auc_mean, auc_std, auc_oof = train(args, script_name, configs, logger)
    prec_time = time.time() - t0
    send_line_notification(f'Finished: {script_name} '
                           f'using CONFIG: {CONFIG_FILE} '
                           f'w/ AUC {auc_mean:.5f}+-{auc_std:.5f} '
                           f'w/ AUC_OOF {auc_oof:.5f} '
                           f'in {prec_time:.1f} s !')
