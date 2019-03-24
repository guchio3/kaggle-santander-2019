import pandas as pd

# load dfs
print('now loading train and test dfs ...')
trn_df = pd.read_csv('./mnt/inputs/origin/train.csv.zip', compression='zip')
tst_df = pd.read_csv('./mnt/inputs/origin/test.csv.zip', compression='zip')

# save idxes
print('now saving train & test index ...')
trn_df['ID_code'].to_pickle(
    './mnt/inputs/nes_info/trn_index.pkl.gz',
    compression='gzip')
tst_df['ID_code'].to_pickle(
    './mnt/inputs/nes_info/tst_index.pkl.gz',
    compression='gzip')

# save target
print('now saving target ...')
trn_df['target'].to_pickle(
    './mnt/inputs/nes_info/target.pkl.gz',
    compression='gzip')
