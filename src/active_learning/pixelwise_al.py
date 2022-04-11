import logging
import os
import time
from collections import OrderedDict

import buzzard as buzz
import click
import cv2 as cv
import git
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch import jit, nn
from tqdm import tqdm

from src.configs import config_factory
from src.loaders.loaders import GTDataset, NoAnnotsRGBDataset
from src.models import NETS
from src.trainer import Trainer
from src.utils.metrics import IoU, accuracy
from src.utils.uncertainty import entropy





class Comparator(Trainer):
    def __init__(self, cfg, net_file, dataset, confidnet_file):
        self.cfg = cfg
        self.device = self._set_device()
        print("Using Device: {}".format(self.device))
        self.net_file = net_file
        logging.info(f"Net file: {net_file}")
        if self.cfg.SAMPLING_STRATEGY == "confidnet":
            self.net = NETS[cfg.NET_NAME](cfg.IN_CHANNELS, cfg.N_CLASSES, cfg.PRETRAIN, confidnet=True).eval().to(self.device)
            torch.manual_seed(7)
            state_dict =jit.load(net_file, map_location=self.device).eval().state_dict()
            self.net.load_state_dict(state_dict)
            self.confidnet = jit.load(confidnet_file, map_location=self.device).eval()
        else:
            self.net = jit.load(net_file, map_location=self.device).eval()
        self.train_dataset = NoAnnotsRGBDataset(dataset, self.cfg, False)
        self.gt_dataset = GTDataset(dataset, self.cfg, self.train_dataset.train_ids)
        logging.info(f"Train ids (len {len(self.train_dataset.imgs)}): {[os.path.basename(i) for i in self.train_dataset.imgs]}"
        )
        self.dataset = dataset
        test_dataset = NoAnnotsRGBDataset(dataset, self.cfg, False)
        logging.info(
            f"Test ids (len {len(test_dataset.imgs)}): {[os.path.basename(i) for i in test_dataset.imgs]}"
        )
        self.metrics = pd.DataFrame(data={i:[] for i in ["n_clicks", "ini_acc", "disir_acc", "disca_acc", "iou_ini", "iou_disir", "iou_disca", "best", "comparator", "area_biggest_error"]})
        self.dataset_name = os.path.basename(dataset.strip('/'))
        if self.cfg.SAMPLING_STRATEGY == "mcdropout":
            self.mc_model = NETS[cfg.NET_NAME](in_channels=cfg.IN_CHANNELS, n_classes=cfg.N_CLASSES, pretrain=cfg.PRETRAIN, drop=True).eval().to(self.device)
            state_dict = self.net.state_dict()
            state_dict =jit.load(net_file, map_location=self.device).eval().state_dict()
            self.mc_model.load_state_dict(state_dict)
            torch.manual_seed(7)  
        self.is_mistake = 0

    def compare(self):
        """Train the network"""
        # Initialization
        logging.info(
            "%s INFO: Begin training",
            time.strftime("%m/%d/%Y %I:%M:%S %p", time.localtime()),
        )
        train_loader = self.train_dataset.get_loader(
            1, self.cfg.WORKERS
        )
        iter_ = 0
        iteru = 0
        steps_pbar = tqdm(
            train_loader, total=self.cfg.EPOCH_SIZE
        )
        confusion_matrixes = np.zeros((3, self.cfg.N_CLASSES, self.cfg.N_CLASSES))
        disir_time = []
        self.disca_time = []
        initial_time = []
        for data in steps_pbar:
            features, labels = data
            output = {}
            features_ = features.clone()
            features_ = torch.cat([features_, torch.zeros(1, self.cfg.N_CLASSES, *features_.shape[-2:])], dim=1)
            features_ = features_.float().to(self.device)
            t = time.time()
            if self.cfg.SAMPLING_STRATEGY == "confidnet":
                output["initial"], _ = self.net(features_)
            else:
                output["initial"] = self.net(features_)
            initial_time.append(time.time()-t)
            annots, raw_annots, n_annots, area, clicked_class = self.simulate_annots(features_, labels.numpy(), output["initial"], iter_)
            if area is None or area < 1:
                iteru +=1
                continue
            features = features.float().to(self.device)
            features = torch.cat([features, annots], dim=1)
            t = time.time()
            if self.cfg.SAMPLING_STRATEGY == "confidnet":
                output["disir"], _ = self.net(features)
            else:
                output["disir"] = self.net(features)
            disir_time.append(time.time()-t)
            output["disca"] = self._disca(features, features_, output["disir"], raw_annots)
            output["disir"] = torch.argmax(output["disir"], dim=1).cpu().numpy()
            output["disca"] = torch.argmax(output["disca"], dim=1).cpu().numpy()
            output["initial"] = torch.argmax(output["initial"], dim=1).cpu().numpy()
            labels = labels.numpy()
            acc_disir = accuracy(output["disir"], labels)
            acc_disca = accuracy(output["disca"], labels)
            acc_ini = accuracy(output["initial"], labels)
            label_comp = torch.Tensor([torch.argmax(torch.Tensor([acc_ini, acc_disir, acc_disca]))]).to(self.device)
            iou_disir = IoU(output["disir"], labels, self.cfg.N_CLASSES)
            iou_disca = IoU(output["disca"], labels, self.cfg.N_CLASSES)
            iou_ini = IoU(output["initial"], labels, self.cfg.N_CLASSES)
            self.metrics.loc[iter_, ["n_clicks", "ini_acc", "disir_acc", "disca_acc", "iou_ini", "iou_disir", "iou_disca", "best", "area_biggest_error", "clicked_class"]] = [n_annots, acc_ini, acc_disir, acc_disca, iou_ini, iou_disir, iou_disca, int(label_comp), area, clicked_class]
            iter_ += 1
            csv_name = "{}_{}{}.csv".format(os.path.join(self.cfg.SAVE_FOLDER, "PixelAL" ), self.dataset_name, f"_{self.cfg.SAMPLING_STRATEGY}")
            self.metrics.to_csv(csv_name, index=False)
            for i, key in enumerate(["initial", "disir", "disca"]):
                confusion = metrics.confusion_matrix(output[key].flatten(), labels.flatten(), labels=np.arange(self.cfg.N_CLASSES))
                confusion_matrixes[i] += confusion
            matrix_name = csv_name.replace(".csv", ".npy")
            np.save(matrix_name, confusion_matrixes)
        logging.info(f"Spotting mistake rate: {self.is_mistake/self.cfg.EPOCH_SIZE}")

    def _disca(self, data, data_, initial_pred, raw_annots):
        lr = self.cfg.CL_LR 
        net = jit.load(self.net_file, map_location=self.device).train()
        
        optimizer = torch.optim.SGD(net.parameters(), lr)
        initial_pred = initial_pred.detach()
        steps = self.cfg.CL_STEPS
        sparse_target = torch.full([*raw_annots.shape[-2:]], -1, dtype=torch.long, device=self.device)
        for i in range(self.cfg.N_CLASSES):
            sparse_target[raw_annots[i]] = i
        tu = time.time()
        sparse_target = sparse_target.unsqueeze(0)
        for iteration in range(steps):
            optimizer.zero_grad()
            pred = net(data_)
            loss = nn.CrossEntropyLoss(ignore_index=-1)(pred, sparse_target)
            reg_fn = nn.L1Loss if self.cfg.CL_REG == "L1" else nn.CrossEntropyLoss
            reg = reg_fn()(pred, initial_pred)
            loss = loss + reg * self.cfg.CL_LAMBDA
            loss.backward()
            optimizer.step()
        output = net(data)
        self.disca_time.append(time.time() - tu)
        return output

    def simulate_annots(self, features, gt, pred_proba, iter_):
        gt = gt[0]
        pred = torch.argmax(pred_proba, dim=1).cpu().numpy()[0]
        pred_proba = torch.nn.Softmax2d()(pred_proba)[0]
        n_annots = self.cfg.BUDGET
        diff_gt_pred = (gt != pred)
        misclassified_classes = np.unique(gt[diff_gt_pred])
        l = []
        for i in misclassified_classes:
            l.append(np.sum(diff_gt_pred[gt == i]))
        if len(misclassified_classes):
            fixed_class = misclassified_classes[np.argmax(l)]
            clicked_class = None
        else:
            fixed_class = -1
            clicked_class = None
        diff_gt_pred[gt != fixed_class] = 0
        sparse_gt = np.zeros_like(gt)
        fp = buzz.Footprint(gt=(0, .1, 0, 10, 0, -.1), rsize=gt.shape)
        polygons = fp.find_polygons(diff_gt_pred)
        order = np.argsort([p.area for p in polygons])
        area = polygons[order[-1]].area if len(order) else None
        if self.cfg.SAMPLING_STRATEGY == "entropy":
            uncertainty = entropy(pred_proba, self.cfg.N_CLASSES)
            uncertainty[diff_gt_pred == 0] = 0
            try:
                quantile = 0.9
                uncertainty = np.where(uncertainty > np.quantile(uncertainty[uncertainty != 0], quantile), 1, 0)
                uncertainty = cv.erode(uncertainty.astype(np.uint8), np.ones((3, 3), dtype=np.uint8))
                polygons_uncertain = fp.find_polygons(uncertainty)
                order_uncertain = np.argsort([p.area for p in polygons_uncertain])
            except:
                order_uncertain = []
        for i in range(min(len(order), n_annots)):
            if self.cfg.SAMPLING_STRATEGY == "max":
                center = fp.spatial_to_raster(np.asarray(polygons[order[-(i+1)]].representative_point().xy).transpose((1, 0)))[0]
                loc = [
                    min(sparse_gt.shape[0] - 1, max(0, center[1]+0)),
                    min(sparse_gt.shape[1] - 1, max(0, center[0]+0)),
                ]
            elif self.cfg.SAMPLING_STRATEGY == "random":
                mistakes = np.where(diff_gt_pred)
                coord = int(torch.randint(len(mistakes[0]),[1]))
                loc = [mistakes[0][coord], mistakes[1][coord]]
            elif self.cfg.SAMPLING_STRATEGY == "entropy":
                if i < len(polygons_uncertain):
                    center = fp.spatial_to_raster(np.asarray(polygons_uncertain[order_uncertain[-(i+1)]].representative_point().xy).transpose((1, 0)))[0]
                    loc = [
                        min(sparse_gt.shape[0] - 1, max(0, center[1]+0)),
                        min(sparse_gt.shape[1] - 1, max(0, center[0]+0)),
                    ]
                else:
                    loc = None
                    n_annots -= 1
            elif self.cfg.SAMPLING_STRATEGY == "mcdropout":
                steps = 5
                moment2 = torch.zeros(self.cfg.N_CLASSES, *features[0, 0].shape).to(self.device)
                mean = torch.zeros(self.cfg.N_CLASSES, *features[0, 0].shape).to(self.device)
                for j in range(steps):
                    pred_ = torch.nn.Softmax2d()(self.mc_model(features))[0].detach()
                    mean += pred_ / steps
                    moment2 += (pred_ ** 2) / steps
                uncertainty = (moment2 - mean ** 2).cpu()
                uncertainty = torch.max(uncertainty,dim=0)[0]
                uncertainty[diff_gt_pred == 0] = 0
                uncertainty = np.where(uncertainty > np.quantile(uncertainty[uncertainty != 0], 0.5), True, False)
                uncertainty = cv.erode(uncertainty.astype(np.uint8), np.ones((3, 3), dtype=np.uint8))

                polygons_uncertain = fp.find_polygons(uncertainty)
                order_uncertain = np.argsort([p.area for p in polygons_uncertain])
                if i < len(polygons_uncertain):
                    center = fp.spatial_to_raster(np.asarray(polygons_uncertain[order_uncertain[-(i+1)]].representative_point().xy).transpose((1, 0)))[0]
                    loc = [
                        min(sparse_gt.shape[0] - 1, max(0, center[1]+0)),
                        min(sparse_gt.shape[1] - 1, max(0, center[0]+0)),
                    ]
                else:
                    loc = None
                    n_annots -= 1
            else:
                raise NotImplementedError(f"sampling strategy {self.cfg.SAMPLING_STRATEGY}")   
            if loc is not None:
                sparse_gt[loc[0], loc[1]] = 1
                self.is_mistake += diff_gt_pred[loc[0], loc[1]]
            clicked_class = gt[loc[0], loc[1]] if loc is not None else None
        raw_annots = [sparse_gt * (gt == i) for i in range(self.cfg.N_CLASSES)]
        raw_annots = np.array(raw_annots).astype(bool)
        if self.cfg.DISTANCE_TRANSFORM:
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
        else:
            kernel = np.ones((3, 3), dtype=np.uint8)
        annots = [cv.dilate(i.astype(np.uint8), kernel)  for i in raw_annots]
        if self.cfg.DISTANCE_TRANSFORM:
            annots = [cv.distanceTransform(i.astype(np.uint8), cv.DIST_L2, 3) for i in annots]
        annots = np.stack(annots)[np.newaxis]
        annots = torch.from_numpy(annots).to(self.device)
        return annots, raw_annots, n_annots, area, clicked_class


cwd = os.getcwd()
print(cwd)


@click.command()
@click.option("-d", "--dataset", help="Dataset on which to train/test.")
@click.option("-c", "--config", help="Path to yaml configuration file.")
@click.option("-p", "--pretrain_file", help="Path to a pretrained network") 
@click.option("--confidnet_file", help="Path to a pretrained network") # "/home/gaston/glowing-garbanzo/data/models/confidnet/ConfidNet_Potsdam_epochs10.pt"

def compare(dataset, config, pretrain_file, confidnet_file):
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
    logging.basicConfig(
        format="%(message)s",
        filename="{}_{}{}.log".format(
            os.path.join(cfg.SAVE_FOLDER, "PixelAL"), dataset_name, f"_{cfg.SAMPLING_STRATEGY}"
            ),
        filemode="w",
        level=logging.INFO,
    )
    logging.info("Git commit: %s", git.Repo().head.object.hexsha)
    logging.info("Config : %s ", cfg)
    logging.info("Dataset, %s", dataset_name)
    logging.info("Pretrained model: %s", pretrain_file)

    comparator = Comparator(cfg, pretrain_file, dataset, confidnet_file)
    comparator.compare()



if __name__ == "__main__":
    compare()
