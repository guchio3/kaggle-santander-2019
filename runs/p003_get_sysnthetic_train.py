import numpy as np
import pandas as pd

from tools.preps.get_nes_info import (get_public_and_private_indices,
                                      get_real_and_synthetic_indices)

print('loading trn_df ...')
trn_df = pd.read_csv('./mnt/inputs/origin/train.csv.zip')

# get real and synthetic_samples_indexes
print('now getting real and synthetic_samples_indexes ...')
real_samples_indexes, synthetic_samples_indexes =\
        get_real_and_synthetic_indices(trn_df)

print('saving ...')
np.save(
    './mnt/inputs/nes_info/trn_real_samples_indexes.npz',
    list(real_samples_indexes))
np.save(
    './mnt/inputs/nes_info/trn_synthetic_samples_indexes.npz',
    list(synthetic_samples_indexes))
