import json
import sys
from os import makedirs

import optuna
import torch
from numpy import pi
from optuna.visualization import *
from torch import nn

import models
from config import Dataset, Config
from main_wlth import get_dataloaders, get_optimizer, iterate_over_dataloader
from models import Model


def main(dataset_name: str, model_name: str):
    study: optuna.Study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, dataset_name, model_name), n_trials=100)
    best_params: dict = study.best_params

    optuna_results_path = 'optuna_results_iter_3'
    optuna_figures_path = 'optuna_figs_iter_3'
    makedirs(optuna_results_path, exist_ok=True)
    makedirs(optuna_figures_path, exist_ok=True)
    run_name = f'{dataset_name}_{model_name}'

    with open(f'{optuna_results_path}/{run_name}.json', 'w') as f:
        json.dump(best_params, f)

    plots_to_save = [
        ('optimization_history', plot_optimization_history),
        ('parallel_coordinate', plot_parallel_coordinate),
        ('contour', plot_contour),
        ('slice', plot_slice),
        ('param_importances', plot_param_importances),
        ('edf', plot_edf),
    ]

    for plot_name, plot_function in plots_to_save:
        fig = plot_function(study, target_name="Accuracy")
        fig.write_image(f'{optuna_figures_path}/{run_name}_{plot_name}.png')


def objective(trial: optuna.Trial, dataset_name: str, model_name: str):
    (adam_lr, weight_decay,
     num_layers, shall_data_reupload, uniform_range) = get_hyperparameters(trial, model_name)

    config = Config(model_name, dataset_name, 16, adam_lr, weight_decay,
                    num_layers, shall_data_reupload, uniform_range)

    dataset = Dataset.get_instance(dataset_name)
    model = Model.get_instance(config, dataset)

    train_dl, valid_dl, test_dl = get_dataloaders(dataset, 16)

    criterion = nn.BCEWithLogitsLoss() if isinstance(model, models.BVQC) else nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, adam_lr, weight_decay)

    for epoch in range(20):
        model.train()
        acc_train, _ = iterate_over_dataloader(model, optimizer, criterion, dataset, train_dl, True)

        with torch.inference_mode():
            model.eval()
            acc_valid, _ = iterate_over_dataloader(model, optimizer, criterion, dataset, valid_dl)

    with torch.inference_mode():
        model.eval()
        acc_test, _ = iterate_over_dataloader(model, optimizer, criterion, dataset, test_dl)

    return acc_test


def get_hyperparameters(trial: optuna.Trial, model_name: str) -> tuple:
    # Adam optimizer
    adam_lr: float = trial.suggest_float('adam_lr', 0.001, 0.3, log=True)
    weight_decay: float = trial.suggest_float('weight_decay', 0.0001, 0.001, log=True)

    # VQC
    if "NN" in model_name:
        return adam_lr, weight_decay, None, None, None

    num_layers: int = trial.suggest_int('num_layers', 8, 16)
    shall_data_reupload: bool = trial.suggest_categorical('shall_data_reupload', [True, False])
    uniform_range: float = trial.suggest_float('uniform_range', -pi, pi, step=0.001)

    return adam_lr, weight_decay, num_layers, shall_data_reupload, uniform_range


if __name__ == '__main__':
    dataset_name = sys.argv[1]
    model_name = sys.argv[2]

    if model_name not in Model.get_names():
        raise ValueError(f"Model \'{model_name}\' not supported.")

    if dataset_name not in Dataset.get_names():
        raise ValueError(f"Dataset \'{dataset_name}\' not supported.")

    if model_name == Model.BVQC_NAME.value and dataset_name in ["3iris", "3wine"]:
        print(f'Dataset \'{dataset_name}\' is not supported for the BVQC model.')

    main(dataset_name, model_name)
