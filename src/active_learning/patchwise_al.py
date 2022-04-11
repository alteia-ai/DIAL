import logging
import os
import time

import buzzard as buzz
import click
import cv2 as cv
import git
import numpy as np
import pandas as pd
import torch
from torch import jit, nn
from tqdm import tqdm
from time import time

from src.configs import config_factory
from src.loaders.loaders import RGBDataset
from src.active_learning.pixelwise_al import Comparator
from src.utils.uncertainty import fgsm_attack, entropy
from src.utils.image import (from_coord_to_patch,
                                               grouper, sliding_window)
from src.utils.metrics import IoU, accuracy, f1_score
from src.models import NETS


class ActiveLearner(Comparator):
    def __init__(self, cfg, net_file, dataset, confidnet_file):
        super().__init__(cfg, net_file, dataset, confidnet_file)
        self.test_dataset = RGBDataset(dataset, cfg, True, False)
        self.img_file = os.path.join(self.cfg.SAVE_FOLDER, self.dataset_name, "*.pt")
        self.metrics = pd.DataFrame()

    def _infer_image(self, stride, data, net, confidence=False, T=1):
        """infer for one image. Rewritten for confidnet and odin"""
        with torch.no_grad():
            img = data
            channels = 1 if confidence else self.cfg.N_CLASSES
            pred = torch.zeros(
                img.shape[1:] + (channels,)
            ) 
            occur_pix = torch.zeros(
                (*img.shape[1:], 1), dtype=torch.uint8
            ) 
        for coords in grouper(
                self.cfg.BATCH_SIZE,
                sliding_window(img, step=stride, window_size=self.cfg.WINDOW_SIZE),
            ):
            data_patches = [from_coord_to_patch(img, coords, self.device)]
            data_patches = torch.cat([*data_patches], dim=1).float()
            if self.cfg.SAMPLING_STRATEGY == "confidnet" and confidence:
                outs = self.confidnet(net(data_patches)[1]).data.cpu()
            else:
                outs = net(data_patches).data.cpu()

            for out, (x, y, w, h) in zip(outs, coords):
                out = out.permute((1, 2, 0))
                pred[x : x + w, y : y + h] += out
                occur_pix[x : x + w, y : y + h, :] += 1
        pred = pred / occur_pix
        pred = pred / T
        if not confidence:
            pred = torch.unsqueeze(pred.permute((2, 0, 1)), 0)
            pred = torch.nn.Softmax2d()(pred)[0]
        else:
            pred = torch.squeeze(pred, -1)
        return pred

    def create_adversary_input(self, stride, img):
        """for odin"""
        adv_img = torch.zeros(self.cfg.N_CLASSES+self.cfg.IN_CHANNELS, *img.shape[1:]) 
        occur_pix = torch.zeros((1, *img.shape[1:]), dtype=torch.uint8)
        criterion = nn.CrossEntropyLoss()
        for coords in grouper(
                self.cfg.BATCH_SIZE,
                sliding_window(img, step=stride, window_size=self.cfg.WINDOW_SIZE),
            ):
            data_patches = [from_coord_to_patch(img, coords, self.device)]
            data_patches = torch.cat([*data_patches], dim=1).float()
            data_patches.requires_grad = True
            outs = self.net(data_patches)
            labels = torch.argmax(outs, dim=1)
            loss = criterion(outs, labels)
            loss.backward()
            perturbed_data = fgsm_attack(data_patches)
            perturbed_data = perturbed_data.detach().cpu()
            perturbed_data[:,self.cfg.IN_CHANNELS:] = 0
            for out, (x, y, w, h) in zip(perturbed_data, coords):
                adv_img[:, x : x + w, y : y + h] += out
                occur_pix[:, x : x + w, y : y + h] += 1
        adv_img = adv_img / occur_pix
        return adv_img

    def compute_uncertainty(self, pred_proba, data):
        t = time()
        if self.cfg.SAMPLING_STRATEGY == "entropy":
            uncertainty = entropy(pred_proba, self.cfg.N_CLASSES)
        elif self.cfg.SAMPLING_STRATEGY == "mcp":
            uncertainty = 1 - torch.max(pred_proba, dim=0)[0]
        elif self.cfg.SAMPLING_STRATEGY == "random" or self.cfg.SAMPLING_STRATEGY == "reading":
            uncertainty = None
        elif self.cfg.SAMPLING_STRATEGY == "odin":
            perturbed_data = self.create_adversary_input(self.cfg.WINDOW_SIZE[0], data)
            uncertainty = self._infer_image(self.cfg.WINDOW_SIZE[0], perturbed_data, self.net, T=100)
            uncertainty = 1 - torch.max(uncertainty, dim=0)[0]
        elif self.cfg.SAMPLING_STRATEGY == "adv":
            perturbed_data = self.create_adversary_input(self.cfg.WINDOW_SIZE[0], data)
            preturbed_proba_pred = self._infer_image(self.cfg.WINDOW_SIZE[0], perturbed_data, self.net)
            uncertainty = torch.max(torch.abs(preturbed_proba_pred - pred_proba), dim=0)[0]
        elif self.cfg.SAMPLING_STRATEGY == "mcdropout":
            steps = 5
            moment2 = torch.zeros(self.cfg.N_CLASSES, *data[0].shape)
            mean = torch.zeros(self.cfg.N_CLASSES, *data[0].shape)
            for j in range(steps):
                pred_ = self._infer_image(self.cfg.WINDOW_SIZE[0], data, self.mc_model).detach()
                mean += pred_ / steps
                moment2 += (pred_ ** 2) / steps
            uncertainty = (moment2 - mean ** 2).cpu()
            uncertainty = torch.max(uncertainty,dim=0)[0]
        elif self.cfg.SAMPLING_STRATEGY == "confidnet":
            uncertainty = self._infer_image(self.cfg.WINDOW_SIZE[0], data, self.net, confidence=True).detach()
            uncertainty = 1 - uncertainty
        else:
            raise NotImplementedError()
        elapsed_time = time() - t
        logging.info(f"Time to compute uncertainty: {elapsed_time}")
        return uncertainty

    def sample_annots(self, pred, gt, pprint):
        sparse_gt = torch.zeros_like(gt)
        diff_gt_pred = (gt != pred)
        fp = buzz.Footprint(gt=(0, .1, 0, 10, 0, -.1), rsize=gt.shape[::-1])
        polygons = fp.find_polygons(diff_gt_pred.numpy())
        order = np.argsort([p.area for p in polygons])
        area = polygons[order[-1]].area if len(order) else None
        for i in range(self.cfg.BUDGET):
            if len(polygons) > i: 
                center = fp.spatial_to_raster(np.asarray(polygons[order[-(i+1)]].representative_point().xy).transpose((1, 0)))[0]
                loc = [
                        min(sparse_gt.shape[0] - 1, max(0, center[1]+0)),
                        min(sparse_gt.shape[1] - 1, max(0, center[0]+0)),
                    ]
                sparse_gt[loc[0], loc[1]] = 1
                if pprint:
                    print(loc)
            else:
                return None, None, None
        raw_annots = [(sparse_gt * (gt == i)).numpy() for i in range(self.cfg.N_CLASSES)]
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
        return annots, raw_annots, area

    def order_tiles(self, fp, tiles, uncertainty_map):
        if self.cfg.SAMPLING_STRATEGY == "random":
            order = torch.randperm(len(tiles))
        elif self.cfg.SAMPLING_STRATEGY == "reading":
            order = torch.arange(len(tiles))
        else:
            fp_uncertainty = torch.zeros(len(tiles))
            for i, tile in enumerate(tiles):
                partial_patch = uncertainty_map[tile.slice_in(fp)[0], tile.slice_in(fp)[1]]
                fp_uncertainty[i] = torch.sum(partial_patch)
            order = torch.argsort(fp_uncertainty, descending=True)
            _ = torch.randperm(len(tiles))
        return order

    def update_weights(self, data, raw_annots, initial_pred, iter_):
        add = 1 
        lr = self.cfg.CL_LR /(iter_ + add)
        self.net.train()
        optimizer = torch.optim.SGD(self.net.parameters(), lr)
        initial_pred = initial_pred.detach()
        steps = self.cfg.CL_STEPS
        sparse_target = torch.full([*raw_annots.shape[-2:]], -1, dtype=torch.long, device=self.device)
        for i in range(self.cfg.N_CLASSES):
            sparse_target[raw_annots[i]] = i
        sparse_target = sparse_target.unsqueeze(0)
        data_ = data.clone()
        initial_pred = initial_pred.to(self.device)
        for iteration in range(steps):
            data_ = data.clone()
            optimizer.zero_grad()
            if torch.rand(1).item() < 0.5: 
                data_[:, self.cfg.IN_CHANNELS:] = 0
            pred = self.net(data_)
            loss = nn.CrossEntropyLoss(ignore_index=-1)(pred, sparse_target)
            reg_fn = nn.L1Loss if self.cfg.CL_REG == "L1" else nn.CrossEntropyLoss
            reg = reg_fn()(pred, initial_pred)
            loss = loss + reg * self.cfg.CL_LAMBDA
            loss.backward()
            optimizer.step()
        self.net.eval()
        return

    def process_img(self, img, gt, iter_):
        filename = self.img_file.replace('*', f"{iter_}")
        if self.cfg.AL_DISCA or self.cfg.SAMPLING_STRATEGY == "confidnet":
            self.net = jit.load(self.net_file, map_location=self.device)
        logging.info(f"Processing file {filename}")
        timer = time()
        data = img[0]
        pred_proba = self._infer_image(self.cfg.WINDOW_SIZE[0], data, self.net)
        pred = torch.argmax(pred_proba, dim=0)
        if self.cfg.SAMPLING_STRATEGY == "confidnet": 
            seed = torch.get_rng_state()
            self.net = NETS[self.cfg.NET_NAME](self.cfg.IN_CHANNELS, self.cfg.N_CLASSES, self.cfg.PRETRAIN, confidnet=True).eval().to(self.device)
            torch.set_rng_state(seed)
            state_dict =jit.load(self.net_file, map_location=self.device).eval().state_dict()
            self.net.load_state_dict(state_dict)
        self.compute_metrics(pred, gt, None, iter_, 0, None)
        new_pred = pred.clone()
        uncertainty_map = self.compute_uncertainty(pred_proba, data)
        logging.info(f"Time pred+uncertainty: {time() - timer}")
        if self.cfg.SAMPLING_STRATEGY == "confidnet":
            self.net = jit.load(self.net_file, map_location=self.device)
        fp = buzz.Footprint(gt=(0, .1, 0, 10, 0, -.1), rsize=(img.shape[-2], img.shape[-1]))
        tiles = fp.tile((self.cfg.WINDOW_SIZE[0], self.cfg.WINDOW_SIZE[1]), boundary_effect="exclude")
        shape = [self.cfg.WINDOW_SIZE[0], self.cfg.WINDOW_SIZE[1]]
        tiles = np.array(tiles).ravel()
        order = self.order_tiles(fp, tiles, uncertainty_map)
        tiles = tiles[order]
        len_tiles = len(tiles)
        if self.cfg.AL_DISCA:
            for i in tqdm(range(len_tiles-1), total=len_tiles-1):
                tile = tiles[0]
                sub_img = img[..., tile.slice_in(fp)[0], tile.slice_in(fp)[1]].clone()
                sub_gt = gt[..., tile.slice_in(fp)[0], tile.slice_in(fp)[1]]
                sub_pred = new_pred[..., tile.slice_in(fp)[0], tile.slice_in(fp)[1]].clone()
                pprint = False
                annots, raw_annots, area = self.sample_annots(sub_pred, sub_gt, pprint)
                if annots is None:
                    self.compute_metrics(new_pred, gt, 0, iter_, i+1, None)
                    tiles = tiles[1:]
                    order = self.order_tiles(fp, tiles, uncertainty_map)
                    tiles = tiles[order]
                    continue
                sub_img[:, 3:] = annots 
                seed = torch.get_rng_state()
                seed_np = np.random.get_state()
                self.update_weights(sub_img, raw_annots, sub_pred, i)
                torch.set_rng_state(seed)
                np.random.set_state(seed_np)
                data[..., tile.slice_in(fp)[0], tile.slice_in(fp)[1]] = torch.unsqueeze(sub_img, 0)
                new_pred_proba = self._infer_image(self.cfg.WINDOW_SIZE[0], data, self.net)
                new_pred = torch.argmax(new_pred_proba, dim=0)
                if uncertainty_map is not None:
                    fp_uncertainty = uncertainty_map[tile.slice_in(fp)[0], tile.slice_in(fp)[1]]
                else:
                    fp_uncertainty = None
                self.compute_metrics(new_pred, gt, area, iter_, i+1, fp_uncertainty)
        else:
            for i, tile in enumerate(tiles):
                sub_img = img[..., tile.slice_in(fp)[0], tile.slice_in(fp)[1]].clone()
                sub_gt = gt[..., tile.slice_in(fp)[0], tile.slice_in(fp)[1]]
                sub_pred = pred[..., tile.slice_in(fp)[0], tile.slice_in(fp)[1]].clone()
                pprint = False
                annots, raw_annots, area = self.sample_annots(sub_pred, sub_gt, pprint)
                if annots is None:
                    self.compute_metrics(new_pred, gt, 0, iter_, i+1, None)
                    continue
                sub_img[:, 3:] = annots 
                reshape_sub_img = torch.zeros(sub_img.shape[0], sub_img.shape[1], self.cfg.WINDOW_SIZE[0], self.cfg.WINDOW_SIZE[1]).to(self.device)
                reshape_sub_img[:,:, :sub_img.shape[2], :sub_img.shape[3]] = sub_img
                out = self.net(reshape_sub_img)[0]
                small_pred = torch.argmax(out, dim=0)
                small_pred = small_pred[:sub_img.shape[2], :sub_img.shape[3]]
                small_sub_slice = tile.slice_in(fp, clip=True)
                big_sub_slice = fp.slice_in(tile, clip=True)
                new_pred[small_sub_slice[0], small_sub_slice[1]] = small_pred[big_sub_slice[0], big_sub_slice[1]]

                if uncertainty_map is not None:
                    fp_uncertainty = uncertainty_map[tile.slice_in(fp)[0], tile.slice_in(fp)[1]]
                else:
                    fp_uncertainty = None
                self.compute_metrics(new_pred, gt, area, iter_, i+1, fp_uncertainty)
        elapsed_time = time() - timer
        logging.info(f"End processing file. {elapsed_time} s")

    def compute_metrics(self, pred, gt, area, iter_, i, uncertainty):
        pred = pred.numpy()
        gt = gt.numpy()
        iou = IoU(pred, gt, self.cfg.N_CLASSES)
        if uncertainty is not None:
            print(f"Step {i}: IoU: {iou}; Uncertain: {uncertainty.mean()}")
        else:
            print(f"Step {i}: IoU: {iou}")
        acc = accuracy(pred, gt)
        f1 = f1_score(pred,gt, self.cfg.N_CLASSES)
        self.metrics.loc[iter_, f"{i}_IoU"] = iou
        self.metrics.loc[iter_, f"{i}_acc"] = acc
        self.metrics.loc[iter_, f"{i}_F1"] = f1
        if area is not None:
            self.metrics.loc[iter_, f"{i}_area"] = area
        if uncertainty is not None:
            self.metrics.loc[iter_, f"{i}_uncertainty_mean"] = float(uncertainty.mean())
            self.metrics.loc[iter_, f"{i}_uncertainty_max"] = float(uncertainty.max())
            self.metrics.loc[iter_, f"{i}_uncertainty_median"] = float(uncertainty.median())
        csv_name = "{}_{}{}.csv".format(
            os.path.join(self.cfg.SAVE_FOLDER, "PatchAL" ), self.dataset_name, f"_ALDISCA{self.cfg.AL_DISCA}_{self.cfg.SAMPLING_STRATEGY}", 
            )
        self.metrics.to_csv(csv_name)

    def run(self):
        loader = self.test_dataset.get_loader(1, self.cfg.WORKERS)
        for iter_, data in enumerate(tqdm(loader, total=len(loader))):
            features, gt = data 
            features = torch.cat([features, torch.zeros(1, self.cfg.N_CLASSES, *features.shape[-2:])], dim=1)
            features = features.to(self.device).float()
            gt = gt[0]
            self.process_img(features, gt, iter_)
cwd = os.getcwd()
print(cwd)


@click.command()
@click.option("-d", "--dataset", help="Dataset on which to train/test.")
@click.option("-c", "--config", help="Path to yaml configuration file.")
@click.option("-p", "--pretrain_file", help="Path to a pretrained network")
@click.option("--confidnet_file", help="Path to a pretrained network")

def run(dataset, config, pretrain_file, confidnet_file):
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
            os.path.join(cfg.SAVE_FOLDER, "PatchAL" ), dataset_name, f"_ALDISCA{cfg.AL_DISCA}_{cfg.SAMPLING_STRATEGY}"
            ),
        filemode="w",
        level=logging.INFO,
    )
    logging.info("Git commit: %s", git.Repo().head.object.hexsha)
    logging.info("Config : %s ", cfg)
    logging.info("Dataset, %s", dataset_name)
    logging.info("Pretrained model: %s", pretrain_file)

    runner = ActiveLearner(cfg, pretrain_file, dataset, confidnet_file)
    runner.run()


if __name__ == "__main__":
    run()