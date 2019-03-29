import numpy as np
import pandas as pd

from tools.preps.get_nes_info import (get_public_and_private_indices,
                                      get_real_and_synthetic_indices)

tst_df = pd.read_csv('./mnt/inputs/origin/test.csv.zip')

# get real and synthetic_samples_indexes
print('now getting real and synthetic_samples_indexes ...')
real_samples_indexes, synthetic_samples_indexes = get_real_and_synthetic_indices(
    tst_df)

print('now getting public and private ...')
public_LB, private_LB = get_public_and_private_indices(
    tst_df, real_samples_indexes, synthetic_samples_indexes)

print('saving ...')
np.save('./mnt/inputs/nes_info/public_LB.npz', list(public_LB))
np.save('./mnt/inputs/nes_info/private_LB.npz', list(private_LB))
np.save(
    './mnt/inputs/nes_info/real_samples_indexes.npz',
    list(real_samples_indexes))
np.save(
    './mnt/inputs/nes_info/synthetic_samples_indexes.npz',
    list(synthetic_samples_indexes))
