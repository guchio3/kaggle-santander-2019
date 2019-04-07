from tools.preps.oof_features import mk_oof_features
from tools.utils.args import parse_mk_oof_feature_args


def main():
    args = parse_mk_oof_feature_args()
    oof_features = mk_oof_features(
        args.oof_filename,
        args.sub_filename,
        args.col_name)
    feature_name = f'./mnt/inputs/features/{args.col_name}.pkl.gz'
    print(f'now saving to {feature_name}')
    oof_features[args.col_name].reset_index(drop=True).to_pickle(feature_name)


if __name__ == '__main__':
    main()
