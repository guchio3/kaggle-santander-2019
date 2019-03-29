import numpy as np
import pandas as pd
from tqdm import tqdm


def get_basic_nes_info():
    # load dfs
    print('now loading train and test dfs ...')
    trn_df = pd.read_csv(
        './mnt/inputs/origin/train.csv.zip',
        compression='zip')
    tst_df = pd.read_csv('./mnt/inputs/origin/test.csv.zip', compression='zip')

    # save idxes
    print('now saving train & test index ...')
    trn_df['ID_code'].to_pickle(
        './mnt/inputs/nes_info/trn_ID_code.pkl.gz',
        compression='gzip')
    tst_df['ID_code'].to_pickle(
        './mnt/inputs/nes_info/tst_ID_code.pkl.gz',
        compression='gzip')

    # save target
    print('now saving target ...')
    trn_df['target'].to_pickle(
        './mnt/inputs/nes_info/target.pkl.gz',
        compression='gzip')


def get_real_and_synthetic_indices(df_test):
    df_test = df_test.drop(['ID_code'], axis=1)
    df_test = df_test.values

    unique_count = np.zeros_like(df_test)
    for feature in tqdm(range(df_test.shape[1])):
        _, index_, count_ = np.unique(
            df_test[:, feature], return_counts=True, return_index=True)
        unique_count[index_[count_ == 1], feature] += 1

    # Samples which have unique values are real the others are fake
    real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
    synthetic_samples_indexes = np.argwhere(
        np.sum(unique_count, axis=1) == 0)[:, 0]
    print(f'REAL_SAMPLES_INDEXES: {len(real_samples_indexes)}')
    print(f'SYNTHETIC_SAMPLES_INDEXES: {len(synthetic_samples_indexes)}')
    return real_samples_indexes, synthetic_samples_indexes


def get_public_and_private_indices(
        df_test, real_samples_indexes, synthetic_samples_indexes):
    ITER = 20000
    df_test = df_test.drop(['ID_code'], axis=1)
    df_test = df_test.values
    df_test_real = df_test[real_samples_indexes].copy()

    generator_for_each_synthetic_sample = []
    # Using 20,000 samples should be enough.
    # You can use all of the 100,000 and get the same results (but 5 times
    # slower)
    for cur_sample_index in tqdm(synthetic_samples_indexes[:ITER]):
        # target の synthetic_samples をとってくる
        cur_synthetic_sample = df_test[cur_sample_index]
        # ↑ の target の各要素と real データの各要素をマッチング
        potential_generators = df_test_real == cur_synthetic_sample

        # A verified generator for a synthetic sample is achieved
        # only if the value of a feature appears only once in the
        # entire real samples set
        # fake のある行と同じ feature が real において 1 つしかないところを feature mask としてとってきている
        # つまりマッチングが１つだけだった列を持ってくる
        features_mask = np.sum(potential_generators, axis=0) == 1
        # マッチングが１つだけだった行をとってくる
        # .sum とかでも良さそうだが、bool にしたかったから any?
        verified_generators_mask = np.any(
            potential_generators[:, features_mask], axis=1)
        verified_generators_for_sample = real_samples_indexes[np.argwhere(
            verified_generators_mask)[:, 0]]
        generator_for_each_synthetic_sample.append(
            set(verified_generators_for_sample))
        public_LB = generator_for_each_synthetic_sample[0]

    # かぶる場合は同じデータセット (public, private)　というロジックで iteration
    for x in tqdm(generator_for_each_synthetic_sample):
        # &
        if public_LB.intersection(x):
            public_LB = public_LB.union(x)

    private_LB = generator_for_each_synthetic_sample[1]
    for x in tqdm(generator_for_each_synthetic_sample):
        # &
        if private_LB.intersection(x):
            private_LB = private_LB.union(x)
    print(f'PUBLIC_LB: {len(public_LB)}')
    print(f'PRIVATE_LB: {len(private_LB)}')
    return public_LB, private_LB
