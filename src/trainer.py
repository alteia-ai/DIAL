import logging
import os
import time
from typing import List

import GPUtil
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.jit as jit

from src.models import NETS


class Trainer:

    def __init__(self, cfg, disir):
        self.number_gpus = 1
        self.disir = disir
        self.cfg = cfg
        self.device = torch.device(self._set_device())
        print("Using Device: {}".format(self.device))

        params = {}
        if cfg.DROPOUT:
            params["drop"] = True
        self.net = self._get_net(cfg.NET_NAME)(
            in_channels=cfg.IN_CHANNELS,
            n_classes=cfg.N_CLASSES,
            pretrain=cfg.PRETRAIN,
            disir=disir,
            **params,
        )
        self.net = self.net.to(self.device)
        self.net_name = self.cfg.NET_NAME
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.OPTIM_BASELR * self.number_gpus,
            momentum=0.9,
            weight_decay=0.0005,
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, list(self.cfg.OPTIM_STEPS), gamma=0.1
        )

    def train(self, epochs, train_ids, test_ids, means, stds, pretrain_file=None):
        pass

    def test(self, test_ids, means, stds, sparsity=0, stride=None):
        pass

    def _get_net(self, net_name: str) -> torch.Tensor:
        return NETS[net_name]

    @staticmethod
    def _set_device():
        """Set gp device when cuda is activated. If code runs with mpi, """
        if not torch.cuda.is_available():
            return "cpu"
        device = "cuda:"
        for d, i in enumerate(GPUtil.getGPUs()):
            if i.memoryUsed < 4500:  # ie:  used gpu memory<900Mo
                device += str(d)
                break
            else:
                if d + 1 == len(GPUtil.getGPUs()):
                    print("\033[91m \U0001F995  All GPUs are currently used by external programs. Using cpu. \U0001F996  \033[0m")
                    device = "cpu"
        return device

    def _save_net(self, epoch: int, accu, iou, train_loss, test_loss, losses, temp=True) -> None:
        state = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "state_dict": self.net.state_dict(),
            "train_loss": train_loss,
            "test_loss": test_loss,
            "epoch": epoch + 1,
            "losses": losses,
            "accu": accu,
            "iou": iou,
        }
        path = os.path.join(self.cfg.PATH_MODELS, "temp") if temp else self.cfg.PATH_MODELS
        if not os.path.isdir(path):
            os.mkdir(path)
        dataset = os.path.basename(self.dataset)
        
        torch.save(
            state, "_".join([os.path.join(path, self.net_name), dataset, f"epoch{epoch}.pth"])
        )

    def load_weights(self, path_weights: str, is_jit=False) -> None:
        """Only to infer (doesn't load scheduler and optimizer state)."""
        print(path_weights)
        if is_jit:
            try:
                self.net = jit.load(path_weights, map_location=self.device)
            except:
                checkpoint = torch.load(path_weights)
                # self.net = jit.load(net_filename, map_location=self.device)
                self.net.load_state_dict(checkpoint["net"])
        else:
            checkpoint = torch.load(path_weights, map_location=str(self.device))
            self.net.load_state_dict(checkpoint["state_dict"])
        
        logging.info("%s Weights loaded", time.strftime("%m/%d/%Y %I:%M:%S %p", time.localtime()))
        print("Weights succesfully loaded")

    def save_to_jit(self, name, is_disir):
        """
        Save the graph to be able to load the network without knowing its architecture.
        Source https://pytorch.org/tutorials/advanced/cpp_export.html#depending-on-libtorch-and-building-the-application
        """
        self.net = self.net.to("cpu")
        n_channels = self.cfg.IN_CHANNELS if not is_disir else self.cfg.IN_CHANNELS + self.cfg.N_CLASSES
        dummy_tensor = torch.randn((self.cfg.BATCH_SIZE, n_channels, *self.cfg.WINDOW_SIZE))
        self.net.eval()
        torch_out = self.net(dummy_tensor)
        traced_script_module = torch.jit.trace(self.net, dummy_tensor)
        traced_script_module.save(name)
        self.net.to(self.device)
        return torch_out, dummy_tensor
