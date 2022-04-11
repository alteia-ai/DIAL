"""
Train a network.
"""
import logging
import os
from os.path import join

import click
import torch
from numpy import random

import git
from src.classic_trainer import ClassicTrainer
from src.configs import config_factory

cwd = os.getcwd()
print(cwd)


@click.command()
@click.option("-d", "--dataset", help="Dataset on which to train/test.")
@click.option(
    "-c", "--config", default=None, help="Path to yaml configuration file.", required=False
)
@click.option(
    "-p", "--pretrain_file", required=False, help="Path to a pretrained network", default=None
)
@click.option("--train/--no-train", help="Activate to skip training (ie only to test)", default=True)


def train(dataset, config, pretrain_file, train):
    """Train a semantic segmentation network on GIS datasets."""
    # Set seeds for reproductibility
    cfg = config_factory(config)
    random.seed(42)
    torch.manual_seed(7)
    torch.backends.cudnn.deterministic = True

    output = cfg.SAVE_FOLDER
    model = cfg.PATH_MODELS

    if not os.path.exists(output):
        os.makedirs(output)

    if not os.path.exists(model):
        os.makedirs(model)

    # Set logger
    # dataset = dataset.strip('/').split("/")[-1]
    print(dataset)
    dataset_name = os.path.basename(dataset.strip('/'))
    logging.basicConfig(
        format="%(message)s",
        filename="{}_{}.log".format(join(cfg.SAVE_FOLDER, cfg.NET_NAME), dataset_name),
        filemode="w",
        level=logging.INFO,
    )
    logging.info("Git commit: %s", git.Repo().head.object.hexsha)
    logging.info("Config : %s ", cfg)
    logging.info("Dataset, %s", dataset_name)
    logging.info("Pretrained model: %s", pretrain_file)

    net = ClassicTrainer(cfg, cfg.DISIR, dataset=dataset)

    if train:
        net.train(cfg.EPOCHS, pretrain_file=pretrain_file)
    elif not cfg.DISIR and pretrain_file:
        net.load_weights(pretrain_file, is_jit=True)
        net.test()

    if cfg.DISIR:
        steps = cfg.N_CLICKS
        id_class = 0
        classes = cfg.N_CLASSES
        if not train and pretrain_file:
            net.load_weights(pretrain_file, is_jit=True)

        net.test(steps, use_previous_state=True, initial_file=pretrain_file)


if __name__ == "__main__":
    train()
