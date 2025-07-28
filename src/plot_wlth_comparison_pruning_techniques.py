from copy import deepcopy
from os import makedirs

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import Dataset
from plot_config import color_mapping, categorize_value, FILE_EXTENSIONS
from src.models import Model


FOLDER_NAME_DATAFRAMES: str = "../dataframes/wlth"
FOLDER_NAME_PLOTS: str = "../plots/wlth/comparison"


def main(dataset: str, model: str, file_extension: str):
    df_iterative_main: pd.DataFrame = pd.read_csv(f'{FOLDER_NAME_DATAFRAMES}/{model}_{dataset}_ITERATIVE.csv')
    df_one_shot_main: pd.DataFrame = pd.read_csv(f'{FOLDER_NAME_DATAFRAMES}/{model}_{dataset}_ONE_SHOT.csv')

    subset_values = ['seed_value', 'i_prune_iter', 'i_epoch', 'remaining_weights', "acc_valid"]

    df_one_shot = deepcopy(df_one_shot_main)[subset_values]
    df_iterative = deepcopy(df_iterative_main)[subset_values]

    def is_in_tolerance_range(value, array, tolerance=0.5):
        return any(np.abs(value - a) <= tolerance for a in array)

    unique_one_shot = df_one_shot['remaining_weights'].unique()
    unique_iterative = df_iterative['remaining_weights'].unique()

    common_weights_iterative = [value
                                for value in unique_iterative
                                if is_in_tolerance_range(value, unique_one_shot, tolerance=0.5)]


    common_weights_one_shot = [value
                               for value in unique_one_shot
                               if is_in_tolerance_range(value, common_weights_iterative, tolerance=0.5)]

    if len(common_weights_iterative) > 4:
        common_weights_iterative = common_weights_iterative[:4]
    if len(common_weights_one_shot) > 4:
        common_weights_one_shot = common_weights_one_shot[:4]

    plot_pruning_method(df_iterative, "iterative", common_weights_iterative, file_extension)
    plot_pruning_method(df_one_shot, "one_shot", common_weights_one_shot, file_extension)


def plot_pruning_method(df, pruning_method, common_weights_list, file_extension: str):
    x = "Epoch"
    y = "Accuracy"

    df = df[df['remaining_weights'].isin(common_weights_list)]
    df.rename(columns={'i_epoch': x}, inplace=True)
    df.rename(columns={'acc_valid': y}, inplace=True)
    df['remaining_weights'] = df['remaining_weights'].astype(str)

    value_color_mapping = {str(value): color_mapping[categorize_value(value)] for value in df['remaining_weights']}

    sns.despine()
    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    sns.lineplot(data=df, x=x, y=y, hue='remaining_weights', palette=value_color_mapping, ax=ax)
    ax.set_ylabel('Accuracy' if pruning_method == "iterative" else '')
    modify_ax(ax)

    plt.tight_layout(rect=[.0, .0, 1., 1.])
    plt.savefig(f'{FOLDER_NAME_PLOTS}/{model}_{dataset}_{y}_{pruning_method}.{file_extension}')
    plt.close()


def modify_ax(ax):
    ax.grid(True)
    ax.set_ylim(0.3 - 0.02, 1.02)
    ax.set_xlim(0, 49)
    ax.legend(title='Remaining\nWeights (%)',
              loc='lower right',
              borderaxespad=0.,
              frameon=True)


if __name__ == '__main__':
    makedirs(f'{FOLDER_NAME_PLOTS}/', exist_ok=True)

    for file_extension in FILE_EXTENSIONS:
        for dataset in Dataset.get_names():
            for model in Model.get_names():
                if dataset.startswith("3") and model == "BVQC" or model == "NN":
                    continue

                print(f'{dataset} - {model}')
                try:
                    main(dataset, model, file_extension)
                except:
                    print("Data for one-shot pruning missing")