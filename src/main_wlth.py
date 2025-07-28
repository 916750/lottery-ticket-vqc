import dataclasses
import sys
from os import makedirs
from typing import List, Literal

import numpy as np
import pandas as pd
import torch
import wandb
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy, MeanMetric

import models
from config import Config, Dataset, PruningTechnique, get_config
from models import Model, MaskInfo

RUN_COUNT = 10

ITERATIVE_PRUNING_RATE: float = 0.2
ONE_SHOT_PRUNING_RATES: List[float] = [0.,
                                       1 - 0.513,
                                       1 - 0.211,
                                       1 - 0.07,
                                       1 - 0.036,
                                       1 - 0.019,
                                       1.]


def set_seed(seed_value: int):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)


def exec_exp(config: Config, seeds: range, pruning_technique: PruningTechnique):
    """ Runs one experiment consisting of multiple runs, given by the seed range """
    is_one_shot_pruning = pruning_technique.value == PruningTechnique.ONE_SHOT.value
    is_iterative_pruning = not is_one_shot_pruning

    run_data = []

    for index, seed in enumerate(seeds):
        weights_before_initial_training = None
        weights_after_initial_training = None

        mask_info = None  # iterative pruning only
        last_remaining_weights = -1  # iterative pruning only

        i_prune_iter = 0

        while True:
            print(f'Seed {seed + 1}/{len(seeds)}, Prune iteration {i_prune_iter}')

            config.seed_value = seed
            pruning_rate = ONE_SHOT_PRUNING_RATES[i_prune_iter] if is_one_shot_pruning else ITERATIVE_PRUNING_RATE

            (new_weights_before_initial_training,
             new_weights_after_initial_training,
             remaining_weights, mask_info) = exec_run(config, i_prune_iter, pruning_rate,
                                                      weights_before_initial_training,
                                                      weights_after_initial_training,
                                                      is_iterative_pruning, mask_info, run_data)

            if new_weights_before_initial_training is not None:
                weights_before_initial_training = new_weights_before_initial_training
            if new_weights_after_initial_training is not None:
                weights_after_initial_training = new_weights_after_initial_training

            # break iterative pruning BEFORE saving current results.
            # 1. remaining weights haven't changed => no difference to last run, so saving is obsolete
            # 2. remaining_weights == 1.9 is the smallest value we want to measure
            #    (lowered to 1.0 to catch lower but close values)
            if is_iterative_pruning and (remaining_weights == last_remaining_weights or remaining_weights < 1.):
                break

            # break one-shot pruning AFTER saving last results.
            # pre-calculating if there is a pruning rate for next run. If no, break, but the current run is still valid.
            if is_one_shot_pruning and i_prune_iter + 1 >= len(ONE_SHOT_PRUNING_RATES):
                break

            last_remaining_weights = remaining_weights
            i_prune_iter += 1

    folder_name = "dataframes/wlth"
    makedirs(folder_name, exist_ok=True)
    file_name = f'{config.model}_{config.dataset}_{pruning_technique.value}.csv'

    df = pd.DataFrame(run_data)
    df.to_csv(f'{folder_name}/{file_name}', index=False)


def get_dataloaders(dataset, batch_size, train_split: float = 0.8):
    train_size = int(train_split * len(dataset))
    valid_size = (len(dataset) - train_size) // 2
    test_size = len(dataset) - train_size - valid_size

    train_d, valid_d, test_d = random_split(dataset, [train_size, valid_size, test_size])

    train_dl = DataLoader(train_d, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_dl = DataLoader(valid_d, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_d, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_dl, valid_dl, test_dl


def get_optimizer(model, learning_rate, weight_decay) -> Adam:
    return Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def iterate_over_dataloader(
        model,
        optimizer,
        criterion,
        dataset,
        data_loader: DataLoader,
        shall_execute_optimizer_steps: bool = False
):
    # TODO: ??? Threshold: classifier needs to be calibrated
    task: Literal["binary", "multiclass"] = "binary" if isinstance(criterion, nn.BCEWithLogitsLoss) else "multiclass"
    accuracy_calculator = Accuracy(threshold=0.5, num_classes=dataset.n_classes, task=task)
    loss_calculator = MeanMetric()

    for x, y in data_loader:
        y_pred = model(x)
        y_true = y.float() if isinstance(criterion, nn.BCEWithLogitsLoss) else y.long()
        loss = criterion(y_pred, y_true)

        if shall_execute_optimizer_steps:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_calculator.update(loss.item())
        accuracy_calculator.update(y_pred, y.long())

    return accuracy_calculator.compute().item(), loss_calculator.compute().item()


def exec_run(config_defaults: Config,
             i_prune_iter: int,
             pruning_rate: float,
             weights_before_initial_training: dict | None,
             weights_after_initial_training: dict | None,
             is_iterative_pruning: bool,
             mask_info: MaskInfo,
             run_data: list):
    run = wandb.init(project="qlth",
                     config=dataclasses.asdict(config_defaults),
                     reinit=True)
    if run is None:
        raise Exception("Run initialization failed!")

    config: Config = wandb.config
    print(config)
    set_seed(config.seed_value)

    dataset = Dataset.get_instance(config.dataset)
    model = Model.get_instance(config, dataset)

    # PRUNING SECTION
    if i_prune_iter == 0:
        weights_before_initial_training = model.get_weights_copy()

    else:
        # prepare for pruning
        model.load_weights_copy(weights_after_initial_training)

        if is_iterative_pruning and mask_info:
            # calculate new pruning rate to take only unpruned weights into account
            p_pruned_old = mask_info.n_pruned / mask_info.n_total
            p_to_prune_new = pruning_rate * mask_info.n_unpruned / mask_info.n_total
            pruning_rate = p_pruned_old + p_to_prune_new

        # prune
        model.prune(pruning_rate)
        mask_info: MaskInfo = model.get_mask_information() if is_iterative_pruning else None

        # reset weights to original
        model.load_weights_copy(weights_before_initial_training)

    wandb.watch(model)
    set_seed(config.seed_value)

    train_dl, valid_dl, test_dl = get_dataloaders(dataset, config.batch_size)

    criterion = nn.BCEWithLogitsLoss() if isinstance(model, models.BVQC) else nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, config.lr, config.weight_decay)

    max_acc_valid = 0
    max_acc_step = 0
    max_acc_state_dict = None

    remaining_weights = model.remaining_weights

    for epoch in range(config.epochs):
        epoch_data = dict(wandb.config)

        model.train()
        acc_train, loss_train = iterate_over_dataloader(model, optimizer, criterion, dataset, train_dl,
                                                        shall_execute_optimizer_steps=True)

        with torch.inference_mode():
            model.eval()
            acc_valid, loss_valid = iterate_over_dataloader(model, optimizer, criterion, dataset, valid_dl)

        if acc_valid >= max_acc_valid:
            max_acc_valid = acc_valid
            max_acc_state_dict = model.state_dict()
            max_acc_step = epoch

        print(f"Epoch: {(epoch + 1):5d}"
              f" | Loss_train: {loss_train:0.7f}"
              f" | Acc_train: {acc_train:0.7f}"
              f" | Loss_valid: {loss_valid:0.7f}"
              f" | Acc_valid: {acc_valid:0.7f}"
              f" | remaining_weights: {remaining_weights}")

        epoch_data.update({
            "i_prune_iter": i_prune_iter,
            "i_epoch": epoch,
            "loss_train": loss_train,
            "acc_train": acc_train,
            "acc_valid": acc_valid,
            "loss_valid": loss_valid,
            "remaining_weights": remaining_weights
        })

        wandb.log({
            "loss_train": loss_train,
            "acc_train": acc_train,
            "acc_valid": acc_valid,
            "loss_valid": loss_valid,
            "remaining_weights": remaining_weights,
            "pruning_technique": pruning_technique
        })

        run_data.append(epoch_data)

    # Test late
    # Load best param config
    if max_acc_state_dict is not None:
        model.load_state_dict(max_acc_state_dict)

    with torch.inference_mode():
        model.eval()
        acc_test, loss_test = iterate_over_dataloader(model, optimizer, criterion, dataset, test_dl)

    print(f"Loss_test: {loss_test:0.7f} | "
          f"Acc_test {acc_test:0.7f} | "
          f"Step: {max_acc_step} | ")

    run.summary["acc_test"] = acc_test
    run.summary["loss_test"] = loss_test
    run.summary["test_step"] = max_acc_step

    for row in run_data:
        row.update({
            "acc_test": acc_test,
            "loss_test": loss_test,
            "test_step": max_acc_step
        })

    run.summary["sum_train_params"] = sum(
        [np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])

    for row in run_data:
        row.update({"sum_train_params": sum(
            [np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])})

    run.finish()

    # for one-shot pruning, we only want the weights after the first training iteration, as all following iterations
    #     are based on the first iteration
    # for iterative pruning, we always want the weights of after the last training iteration, as every iteration is
    #     based on the previous one
    shall_save_weights_after_training: bool = i_prune_iter == 0 or is_iterative_pruning
    weights_after_initial_training = model.get_weights_copy() if shall_save_weights_after_training else None

    return weights_before_initial_training, weights_after_initial_training, remaining_weights, mask_info


if __name__ == "__main__":
    config = get_config()
    pruning_technique = PruningTechnique.get_instance(sys.argv[3])

    exec_exp(config, range(RUN_COUNT), pruning_technique)
