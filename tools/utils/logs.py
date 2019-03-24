import functools
import time
from logging import DEBUG, FileHandler, Formatter, StreamHandler

import requests
from lightgbm.callback import _format_eval_result


# ==========================================
#  logging utils
# ==========================================
def logInit(logger, log_dir, log_filename):
    '''
    Init the logger.

    '''
    log_fmt = Formatter('%(asctime)s %(name)s \
            %(lineno)d [%(levelname)s] [%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(log_dir + log_filename, 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)
    return logger


def sel_log(message, logger, debug=False):
    '''
    Use logger if specified one, and use print otherwise.
    Also it's possible to specify to use debug mode (default: info mode).

    The func name is the shorter version of selective_logging.

    '''
    if logger:
        if debug:
            logger.debug(message)
        else:
            logger.info(message)
    else:
        print(message)


def log_evaluation(logger, period=1, show_stdv=True, level=DEBUG):
    def _callback(env):
        if period > 0 and \
                env.evaluation_result_list and \
                (env.iteration + 1) % period == 0:
            result = '\t'.join(
                [_format_eval_result(x, show_stdv)
                 for x in env.evaluation_result_list])
            logger.log(level, '[{}]\t{}'.format(env.iteration + 1, result))
    _callback.order = 10
    return _callback


def dec_timer(func):
    '''
    Decorator which measures the processing time of the func.

    '''
    # wraps func enable to hold the func name
    @functools.wraps(func)
    def _timer(*args, **kwargs):
        t0 = time.time()
        start_str = f'[{func.__name__}] start'
        if 'logger' in kwargs:
            logger = kwargs['logger']
        else:
            logger = None
        sel_log(start_str, logger)

        # run the func
        res = func(*args, **kwargs)

        end_str = f'[{func.__name__}] done in {time.time() - t0:.1f} s'
        sel_log(end_str, logger)
        return res

    return _timer


# ==========================================
#  life hack utils
# ==========================================
def send_line_notification(message):
    line_token = 'yEbRkNfX02oVGlGLI23JCVMHZzp0r2JIVvbaU1NLHIh'
    endpoint = 'https://notify-api.line.me/api/notify'
    message = "\n{}".format(message)
    payload = {'message': message}
    headers = {'Authorization': 'Bearer {}'.format(line_token)}
    requests.post(endpoint, data=payload, headers=headers)
