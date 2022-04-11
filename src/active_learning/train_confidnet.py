import logging
import os
import time

import click
import git
import numpy as np
import torch
from torch import jit, nn, optim
from tqdm import tqdm

from src.configs import config_factory
from src.loaders.loaders import NoAnnotsRGBDataset, RGBDataset
from src.trainer import Trainer
from src.utils.image import (from_coord_to_patch,
                                               grouper, sliding_window)
from src.models.confidnet import ConfidNet
from src.models import NETS


class ConfidNetTrainer(Trainer):
    def __init__(self, cfg, net_file, dataset, name):
        """
        :param cfg: configuration file
        :param interactive: If true, enables interactivity.
        :param *nets_lf: the two trained networks (on RGB and DSM)
            used for late fusion. Order : RGB net then DSM net.
        """
        self.cfg = cfg
        self.device = self._set_device()
        self.net_file = net_file
        logging.info(f"Net file: {net_file}")
        self.train_dataset = NoAnnotsRGBDataset(dataset, self.cfg, True)
        self.test_dataset = RGBDataset(dataset, self.cfg, True, False)
        self.confidnet = ConfidNet().to(self.device)
        lr = 0.0001
        logging.info(f"Learning rate: {lr}")
        self.optimizer = optim.Adam(self.confidnet.parameters(), lr)
        self.name = name
        self.net = NETS[cfg.NET_NAME](cfg.IN_CHANNELS, cfg.N_CLASSES, cfg.PRETRAIN, confidnet=True).eval().to(self.device)
        state_dict =jit.load(net_file, map_location=self.device).eval().state_dict()
        self.net.load_state_dict(state_dict)
        torch.manual_seed(7) 

    def train(self):
        """
        Dans l'implementation originae, il fait une régression L2 en utilisant le softmax du réseau principal
         comme VT: ca me semble faussé puisque on n'est pas sur de ca justement. Ils ne gardent pas la BCE 
         avec la carte binaire (bonne prédiction/fausse prédiction). Je ne comprends pas pourquoi,
          je vais faire comme ca moi.
        """
        epochs = self.cfg.EPOCHS
        for e in tqdm(range(epochs), total=epochs):
            logging.info(
                "\n%s Epoch %s",
                time.strftime("%m/%d/%Y %I:%M:%S %p", time.localtime()),
                e,
            )

            self.confidnet.train()
            loader = self.train_dataset.get_loader(self.cfg.BATCH_SIZE, self.cfg.WORKERS)
            steps_pbar = tqdm(loader, total=self.cfg.EPOCH_SIZE // self.cfg.BATCH_SIZE)
            losses = []
            for data in steps_pbar:
                features, labels = data
                self.optimizer.zero_grad()
                features = torch.cat([features, torch.zeros(features.shape[0], self.cfg.N_CLASSES, *features.shape[-2:])], dim=1)
                features = features.float().to(self.device)
                labels = labels.float().to(self.device)
                pred_proba, feat = self.net(features)
                output = self.confidnet(feat)
                output = torch.squeeze(output, dim=1)
                pred_proba = torch.nn.Softmax2d()(pred_proba)
                one_hot_labels = torch.stack([labels==i for i in range(self.cfg.N_CLASSES)], dim=1)
                gt = torch.sum(pred_proba * one_hot_labels, dim=1)
                loss = nn.MSELoss()(output, gt)

                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())     
                steps_pbar.set_postfix({"loss": loss.item()})
            logging.info(f"Training loss: {np.mean(losses)}")
            self.save_to_jit()
            self.eval()
        self.save_to_jit()

    def save_to_jit(self):
        name = self.name.replace(self.cfg.SAVE_FOLDER, self.cfg.PATH_MODELS) + "pt"
        self.confidnet = self.confidnet.to("cpu").eval()
        dummy_tensor = torch.randn((self.cfg.BATCH_SIZE, 32, *self.cfg.WINDOW_SIZE))
        torch_out = self.confidnet(dummy_tensor)
        traced_script_module = torch.jit.trace(self.confidnet, dummy_tensor)
        traced_script_module.save(name)
        self.confidnet.to(self.device)

    def _evaluate_image(self, img, labels):
        """infer for one image"""
        with torch.no_grad():
            pred = np.zeros(
                img.shape[1:] + (self.cfg.N_CLASSES,)
            ) 
        accuracy = []
        l2 = []
        for coords in grouper(
                self.cfg.BATCH_SIZE,
                sliding_window(img, step=self.cfg.WINDOW_SIZE[0], window_size=self.cfg.WINDOW_SIZE),
            ):
            data_patches = [from_coord_to_patch(img, coords, self.device)]
            data_patches = torch.cat([*data_patches], dim=1).float()
            gt_patches = [from_coord_to_patch(labels, coords, self.device)]
            gt_patches = torch.cat([*gt_patches], dim=1).float()
            pred_proba, feat = self.net(data_patches)
            outs = self.confidnet(feat).data
            outs = torch.squeeze(outs, 1)
            pred = torch.argmax(pred_proba, dim=1)
            gt_patches = torch.squeeze(gt_patches, dim=1)
            diff_gt_pred = gt_patches == pred
            accuracy.append(float(torch.mean(((outs > 0.5) == diff_gt_pred).float())))
            pred_proba = torch.nn.Softmax2d()(pred_proba)
            one_hot_labels = torch.stack([gt_patches==i for i in range(self.cfg.N_CLASSES)], dim=1)
            gt = torch.sum(pred_proba * one_hot_labels, dim=1)
            loss = nn.MSELoss()(outs, gt)
            l2.append(loss.item())
        return 100 * np.mean(accuracy), np.mean(l2)


    def eval(self):
        self.confidnet.eval()
        loader = self.test_dataset.get_loader(1, self.cfg.WORKERS)
        steps_pbar = tqdm(loader, total=self.cfg.EPOCH_SIZE // self.cfg.BATCH_SIZE)
        accuracy = []
        l2_loss = []
        for data in steps_pbar:
            features, labels = data
            features = torch.cat([features, torch.zeros(features.shape[0], self.cfg.N_CLASSES, *features.shape[-2:])], dim=1)
            features = features.float().to(self.device)[0]
            labels = labels.float().to(self.device)
            acc, l2 = self._evaluate_image(features, labels)
            accuracy.append(acc)
            l2_loss.append(l2)
        accuracy = np.mean(accuracy)
        l2_loss = np.mean(l2_loss)
        logging.info(f"Evaluation accuracy: {np.round(accuracy, 2)}. L2 loss: {np.round(l2_loss, 2)}.")


cwd = os.getcwd()
print(cwd)


@click.command()
@click.option("-d", "--dataset", help="Dataset on which to train/test.")
@click.option("-c", "--config", help="Path to yaml configuration file.")
@click.option("-p", "--pretrain_file", help="Path to a pretrained network")

def train(dataset, config, pretrain_file):
    """Train a semantic segmentation network on GIS datasets."""
    # Set seeds for reproductibility
    cfg = config_factory(config)
    np.random.seed(42)
    torch.manual_seed(7)
    torch.backends.cudnn.deterministic = True

    output = cfg.SAVE_FOLDER
    model = cfg.PATH_MODELS

    if not os.path.exists(output):
        os.makedirs(output)

    if not os.path.exists(model):
        os.makedirs(model)

    print(dataset)
    dataset_name = os.path.basename(dataset.strip('/'))
    name = os.path.join(cfg.SAVE_FOLDER, f"ConfidNet_{dataset_name}_epochs{cfg.EPOCHS}.")
    logging.basicConfig(
        format="%(message)s",
        filename=name+"log",
        filemode="w",
        level=logging.INFO,
    )
    logging.info("Git commit: %s", git.Repo().head.object.hexsha)
    logging.info("Config : %s ", cfg)
    logging.info("Dataset, %s", dataset_name)
    logging.info("Pretrained model: %s", pretrain_file)

    trainer = ConfidNetTrainer(cfg, pretrain_file, dataset, name)
    trainer.train()



if __name__ == "__main__":
    train()
