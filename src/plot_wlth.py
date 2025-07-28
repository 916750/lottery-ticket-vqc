import time
from os import makedirs
from typing import List

import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from pandas import DataFrame

from config import PruningTechnique
from config import Dataset
from models import Model
from plot_config import color_mapping, categorize_value
from src.plot_config import FILE_EXTENSIONS

FOLDER_NAME_DATAFRAMES: str = "../dataframes/wlth"
FOLDER_NAME_PLOTS: str = "../plots/wlth"

REMAINING_WEIGHTS_LIMIT = 8
GRAPH_Y_LIMITS: List[float] = [0.3]
SHALL_SMOOTH = [False, True]
SELECT_CURVES = [True, False]

subset_values = ['seed_value', 'i_prune_iter', 'i_epoch', 'remaining_weights']
keywords = ["acc_valid",
            # "acc_train",
            # "loss_valid",
            # "loss_train"
            ]

curve_selection = {
    "3iris": {
        "VQC": ["100.0", "26.0", "10.9", "64.1"],
        "SNN": ["100.0", "32.1", "26.2", "13.7"]
    },
    "2iris": {
        "BVQC": ["100.0", "80.0", "33.3", "26.7", "21.7"],
        "VQC": ["100.0", "51.1", "26.1", "21.1", "13.3", "10.6"],
        "SNN": ["100.0", "20.8", "13.2", "10.4"]
    },
    "3wine": {
        "VQC": ["100.0", "80.0", "32.7", "26.1", "4.3"],
        "SNN": ["100.0", "21.1", "41.1", "16.9", "13.5"]
    },
    "2wine": {
        "BVQC": ["100.0", "51.3", "41.0", "32.8"],
        "VQC": ["100.0", "51.3", "41.0", "32.8"],
        "SNN": ["100.0", "51.4", "21.4", "16.9"]
    }
}


def main(dataset: str, model: str, pruning_technique: PruningTechnique, file_extension: str, model_index: int):
    for shall_smooth in SHALL_SMOOTH:
        for graph_y_limit in GRAPH_Y_LIMITS:
            for select_curves in SELECT_CURVES:
                print(f"- options: graph_y_limit: {graph_y_limit}, {'smoothed' if shall_smooth else 'unsmoothed'}, {'selected' if select_curves else 'full'}")

                for keyword in keywords:
                    df = pd.read_csv(f'{FOLDER_NAME_DATAFRAMES}/{model}_{dataset}_{pruning_technique.value}.csv')
                    df = df[subset_values + [keyword]]

                    plot_dataset_grid(model, df, keyword, shall_smooth, graph_y_limit, select_curves, dataset, file_extension, model_index)


def plot_dataset_grid(model: str, df: DataFrame, y, shall_smooth, graph_y_limit, select_curves, dataset, file_extension: str,
                      model_index: int):
    x = "Epoch"
    new_y = "Accuracy"

    seaborn.despine()
    seaborn.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    df.rename(columns={'i_epoch': x}, inplace=True)

    if y == "acc_valid":
        df.rename(columns={'acc_valid': new_y}, inplace=True)

    if select_curves:
        df['remaining_weights'] = df['remaining_weights'].astype(str)
        curves_to_select = curve_selection[dataset][model]
        df = df[df["remaining_weights"].isin(curves_to_select)]
    else:
        df = df[df["remaining_weights"] > REMAINING_WEIGHTS_LIMIT]
        df['remaining_weights'] = df['remaining_weights'].astype(str)

    if shall_smooth:
        y_smooth = f'{new_y}_smooth'
        df[y_smooth] = df[new_y].ewm(span=5).mean()
        overwrite_values = 3
        df[y_smooth].iloc[:overwrite_values] = df[new_y].iloc[:overwrite_values]

    value_color_mapping = {str(value): color_mapping[categorize_value(value)] for value in df['remaining_weights']}

    seaborn.lineplot(data=df, x=x, y=(y_smooth if shall_smooth else new_y),
                     ax=ax,
                     hue="remaining_weights",
                     palette=value_color_mapping)

    ax.grid(True, zorder=0)
    if "acc" in y:
        ax.set_ylim(graph_y_limit - 0.02, 1.02)
    ax.set_ylabel('Accuracy' if model_index == 1 else '')
    ax.set_xlim(0, 49)

    if select_curves:
        legend = ax.legend(title='Remaining\nWeights (%)',
                           loc=('upper left' if model == 'VQC' and dataset == '3wine' else 'lower right'),
                           borderaxespad=0.,
                           frameon=True)
        plt.tight_layout(rect=[.0, .0, 1., 1.])
        plt.subplots_adjust(bottom=0.15)
    else:
        legend = ax.legend(title='Remaining Weights (%)',
                           loc='lower center',
                           borderaxespad=0.,
                           frameon=False,
                           ncol=4,
                           bbox_to_anchor=(0.5, -0.65))
        plt.tight_layout(rect=[.0, .4, 1., 1.])
        plt.subplots_adjust(bottom=0.4)

    legend.set_zorder(5)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_alpha(1.)

    plt.xlabel(x)
    plt.ylabel('Accuracy' if model_index == 1 else '')

    plt.savefig(f'{FOLDER_NAME_PLOTS}/'
                f'{dataset}'
                f'_{y}'
                f'_{model}'
                f'{f"_{graph_y_limit}" if "acc" in y and graph_y_limit > 0.3 else ""}'
                f'{"_smoothed" if shall_smooth else ""}'
                f'{"_selected" if select_curves else ""}'
                f'.{file_extension}')

    plt.close(fig)


if __name__ == '__main__':
    makedirs(f'{FOLDER_NAME_PLOTS}/', exist_ok=True)

    start_time = time.time()

    for file_extension in FILE_EXTENSIONS:
        for pruning_technique in [
            PruningTechnique.ITERATIVE,
            # PruningTechnique.ONE_SHOT
        ]:

            for dataset in Dataset.get_names():
                model_index = 0

                for model in Model.get_names():
                    if dataset.startswith("3") and model == "BVQC" or model == "NN":
                        continue

                    model_index += 1

                    print(f'{dataset} - {model} - {pruning_technique.value} ({file_extension})')
                    main(dataset, model, pruning_technique, file_extension, model_index)

    duration = time.time() - start_time
    print(f'Took {duration // 60} minutes, {duration % 60} seconds')
