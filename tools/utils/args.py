import argparse

from .logs import sel_log


def parse_train_args(logger=None):
    '''
    Policy
    ------------
    * experiment id must be required

    '''
    parser = argparse.ArgumentParser(
        prog='XXX.py',
        usage='ex) python -t -s -m "e010, oversampling"',
        description='short explanation of args',
        add_help=True,
    )
    parser.add_argument('-t', '--test',
                        help='set when you run test',
                        action='store_true',
                        default=False)
    parser.add_argument('-d', '--debug',
                        help='whether or not to use debug mode',
                        action='store_true',
                        default=False)
    parser.add_argument('-s', '--submit',
                        help='submit the prediction',
                        action='store_true',
                        default=False)
    parser.add_argument('-m', '--message',
                        help='messages about the process',
                        type=str,
                        default='')

    args = parser.parse_args()
    if args.submit:
        assert args.test + args.submit > 1, \
            'U cannot submit w/o test prediction.'
    assert args.debug + args.submit < 2, 'U cannot submit w/ debug mode.'
    sel_log(f'args: {sorted(vars(args).items())}', logger)
    return args


def parse_feature_args(logger=None):
    '''
    Policy
    ------------
    * experiment id must be required

    '''
    parser = argparse.ArgumentParser(
        prog='XXX.py',
        usage='ex) python XXX.py -f f001 -m "mk f001 features"',
        description='short explanation of args',
        add_help=True,
    )
    parser.add_argument('-f', '--feature_ids',
                        help='feature id',
                        type=str,
                        nargs='+',
                        required=True)
    parser.add_argument('-m', '--message',
                        help='messages about the process',
                        type=str,
                        default='')

    args = parser.parse_args()
    sel_log(f'args: {sorted(vars(args).items())}', logger)
    return args


def parse_mk_oof_feature_args(logger=None):
    '''
    Policy
    ------------
    * experiment id must be required

    '''
    parser = argparse.ArgumentParser(
        prog='XXX.py',
        usage='ex) python XXX.py -f f001 -m "mk f001 features"',
        description='short explanation of args',
        add_help=True,
    )
    parser.add_argument('-o', '--oof_filename',
                        type=str,
                        required=True)
    parser.add_argument('-s', '--sub_filename',
                        type=str,
                        required=True)
    parser.add_argument('-c', '--col_name',
                        type=str,
                        required=True)

    args = parser.parse_args()
    sel_log(f'args: {sorted(vars(args).items())}', logger)
    return args
