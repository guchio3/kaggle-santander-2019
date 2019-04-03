from itertools import chain

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# ==========================================
#  tools for train info
# ==========================================
def num_feature_comp_hist(features, titles, colors):
    num_features = len(features)
    fig, axs = plt.subplots(1, num_features, figsize=(5 * num_features, 4))
    xmin = np.min([feature.min() for feature in features if len(feature) > 0])
    xmax = np.max([feature.max() for feature in features if len(feature) > 0])
    for feature, title, color, ax in zip(features, titles, colors, axs):
        sns.distplot(feature, kde=False, ax=ax, color=color)
        ax.set_xlim(xmin, xmax)
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


def num_feature_comp_hist_w_target(features, titles, colors):
    num_features = len(features)
    fig, axs = plt.subplots(1, num_features, figsize=(5 * num_features, 4))
    xmin = np.min([feature.min() for feature in features if len(feature) > 0])
    xmax = np.max([feature.max() for feature in features if len(feature) > 0])
    for feature, title, color, ax in zip(features, titles, colors, axs):
        sns.distplot(feature, kde=False, ax=ax, color=color)
        ax.set_xlim(xmin, xmax)
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ==========================================
#  tools for importance visualization
# ==========================================
def save_importance(features, fold_importance_dict,
                    filename_base, topk=30, main_metric='gain'):
    assert main_metric in ['gain', 'split'], \
        f'please specify gain or split as main_metric'
    dfs = []
    for fold in fold_importance_dict:
        df = fold_importance_dict[fold]
        df = df.add_suffix(f'_{fold}')
        dfs.append(df)
    df = pd.concat(dfs, axis=1)
    df['features'] = features
    splits = df.loc[:, df.columns.str.contains('split')]
    gains = df.loc[:, df.columns.str.contains('gain')]

    # stats about splits
    df['split_mean'] = splits.mean(axis=1)
    df['split_std'] = splits.std(axis=1)
    df['split_cov'] = df.split_std / df.split_mean

    # stats about gains
    df['gain_mean'] = gains.mean(axis=1)
    df['gain_std'] = gains.std(axis=1)
    df['gain_cov'] = df.gain_std / df.gain_mean

    # sort and save to csv
    df.sort_values(by=main_metric + '_mean', ascending=False, inplace=True)
    df.to_csv(filename_base + '.csv', index=False)

    # plot and save fig
    plt_dfs = []
    for fold in fold_importance_dict:
        plt_df = pd.DataFrame(fold_importance_dict[fold][main_metric])
        plt_df['features'] = features
        plt_dfs.append(plt_df)
    plt_df = pd.concat(plt_dfs, axis=0)

    # Plot! note that use only top-k
    plt.figure(figsize=(10, 20))
    sns.barplot(x=main_metric, y='features', data=plt_df,
                order=df.features.head(topk))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(filename_base + '.png')
