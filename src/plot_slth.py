from copy import deepcopy
from os import makedirs

import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from pandas import DataFrame

from config import Dataset
from models import Model
from src.plot_config import FILE_EXTENSIONS

FOLDER_NAME_DATAFRAMES: str = "../dataframes/slth"
FOLDER_NAME_DATAFRAMES_WEAK: str = "../dataframes/wlth"
FOLDER_NAME_PLOTS: str = "../plots/slth"

SHALL_SMOOTH = [False, True]

curve_selection = {
    "3iris": {
        "VQC": ["51.0"],
        "SNN": ["40.5"],
    },
    "2iris": {
        "BVQC": ["51.7"],
        "VQC": ["32.8"],
        "SNN": ["41.0"],
    },
    "3wine": {
        "VQC": ["32.7"],
        "SNN": ["41.1"],
    },
    "2wine": {
        "BVQC": ["41.0"],
        "VQC": ["26.2"],
        "SNN": ["41.1"],
    }
}


def main(dataset: str, model: str, file_extension: str, model_index: int):

    for shall_smooth in SHALL_SMOOTH:
        print(f"- options: {'smoothed' if shall_smooth else 'unsmoothed'}")

        df = pd.read_csv(f'{FOLDER_NAME_DATAFRAMES}/{model}_{dataset}_75x25.csv')
        plot_dataset_grid(model, df, shall_smooth, dataset, file_extension, model_index)


def plot_dataset_grid(model: str,
                      df: DataFrame,
                      shall_smooth: bool,
                      dataset: str,
                      file_extension: str,
                      model_index: int):
    df_subset = deepcopy(df[df['i_individual'] == 24])
    df_subset['sparsity'] = 100 - df_subset['sparsity']
    df_subset.rename(columns={'i_gen': 'generation'}, inplace=True)

    # weak lth for comparison
    df_weak = pd.read_csv(f'{FOLDER_NAME_DATAFRAMES_WEAK}/{model}_{dataset}_ITERATIVE.csv')
    df_weak = df_weak[['seed_value', 'i_epoch', 'remaining_weights'] + ["acc_valid"]]
    df_weak['remaining_weights'] = df_weak['remaining_weights'].astype(str)
    df_weak = df_weak[df_weak["i_epoch"].astype(str).isin(["49"])]
    df_weak_100 = df_weak[df_weak["remaining_weights"].isin(["100.0"])]
    df_weak_selected = df_weak[df_weak["remaining_weights"].isin(curve_selection[dataset][model])]


    # smoothing
    if shall_smooth:
        # Smooth accuracy
        y_smooth_acc = 'accuracy_smooth'
        df_subset[y_smooth_acc] = df_subset['accuracy'].ewm(span=3, adjust=False).mean()
        df_subset = df_subset.dropna(subset=[y_smooth_acc])

        # Smooth sparsity
        y_smooth_spars = 'sparsity_smooth'
        df_subset[y_smooth_spars] = df_subset['sparsity'].ewm(span=3, adjust=False).mean()
        df_subset = df_subset.dropna(subset=[y_smooth_spars])

    seaborn.despine()
    seaborn.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # Plot accuracy on the primary y-axis
    seaborn.lineplot(data=df_subset, x='generation', y='accuracy', ax=ax, label='Accuracy')
    ax.set_ylim(0., 1.02)
    ax.set_xlim(1, 75)
    ax.set_yticks([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylabel('Accuracy' if model_index == 1 else '')
    ax.set_xlabel('Generation')
    ax.grid(True, zorder=0)

    # Create a secondary y-axis for sparsity
    ax2 = ax.twinx()
    seaborn.lineplot(data=df_subset, x='generation', y='sparsity', c="orange", ax=ax2, label='Remaining Weights')
    ax2.set_ylim(0, 102)
    ax2.set_yticks(list(range(0, 101, 10)))
    ax2.set_ylabel('Remaining Weights (%)' if (model_index == 2 and dataset.startswith('3')) or model_index == 3 else '')
    ax2.grid(False)

    ax2.axhline(y=df_weak_100["acc_valid"].mean() * 100, color="#A0C4FF", linestyle='dashed', label="100% Remaining Weights")
    ax2.axhline(y=df_weak_selected["acc_valid"].mean() * 100, color="#FFABAB", linestyle='dotted', label=f"{curve_selection[dataset][model][0]}% Remaining Weights")

    ax.legend_.remove()
    ax2.legend_.remove()

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    legend = ax.legend(handles1 + handles2,
                       labels1 + labels2,
                       loc='lower left',
                       borderaxespad=0.,
                       frameon=True)
    legend.set_zorder(10)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_alpha(1.)

    plt.tight_layout(rect=[.0, .0, 1., 1.])
    plt.savefig(f'{FOLDER_NAME_PLOTS}/{dataset}{"_smoothed" if shall_smooth else ""}_{model}.{file_extension}')
    plt.show()


if __name__ == '__main__':
    makedirs(f'{FOLDER_NAME_PLOTS}/', exist_ok=True)

    for file_extension in FILE_EXTENSIONS:
        for dataset in Dataset.get_names():
            model_index = 0

            for model in Model.get_names():
                if dataset.startswith("3") and model == "BVQC" or model == "NN":
                    continue

                model_index += 1

                print(f'{dataset} - {model}')
                main(dataset, model, file_extension, model_index)
