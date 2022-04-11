import logging
import os
import time
from glob import glob

import numpy as np
import pandas as pd
import torch
from scipy.ndimage.morphology import distance_transform_edt
from sklearn import metrics
from torch import nn, optim
from tqdm import tqdm

from src.loaders.loaders import GTDataset, RGBDataset
from src.trainer import Trainer
from src.utils.image import (encode_annots,
                                               from_coord_to_patch, grouper,
                                               sliding_window)
from src.utils.losses import CrossEntropy2d
from src.utils.metrics import IoU, accuracy, f1_score
from src.loaders.sparsifier import Sparsifier


def freeze_bn(module):
    for name, m in module.named_children():
        if len(list(m.children())) > 0:
            freeze_bn(m)
        if "bn" in name: 
            m.weight.requires_grad = False 
            m.bias.requires_grad = False 

class ClassicTrainer(Trainer):
    def __init__(self, cfg, disir, train=True, dataset=None):
        super(ClassicTrainer, self).__init__(cfg, disir)
        if train:
            self.train_dataset = RGBDataset(dataset, self.cfg, self.disir)
            self.gt_dataset = GTDataset(dataset, self.cfg, self.train_dataset.train_ids)
            logging.info(f"Train ids (len {len(self.train_dataset.imgs)}): {[os.path.basename(i) for i in self.train_dataset.imgs]}"
            )
        self.dataset = dataset
        test_dataset = RGBDataset(dataset, self.cfg, self.disir, False)
        logging.info(
            f"Test ids (len {len(test_dataset.imgs)}): {[os.path.basename(i) for i in test_dataset.imgs]}"
        )
        self.metrics = pd.DataFrame(data={i:[] for i in [os.path.basename(i) for i in test_dataset.imgs]}).T
 

    def train(self, epochs, pretrain_file=None):
        """Train the network"""
        #  Initialization
        logging.info(
            "%s INFO: Begin training",
            time.strftime("%m/%d/%Y %I:%M:%S %p", time.localtime()),
        )

        iter_ = 0

        start_epoch, accu, iou, f1, train_loss, test_loss, losses = self._load_init(
            pretrain_file
        )
        loss_weights = torch.ones(
            self.cfg.N_CLASSES, dtype=torch.float32, device=self.device
        )
        if self.cfg.WEIGHTED_LOSS:
            weights = self.gt_dataset.compute_frequency()
            if self.cfg.WEIGHTED_LOSS:
                loss_weights = (
                    torch.from_numpy(weights).type(torch.FloatTensor).to(self.device)
                )

        train_loader = self.train_dataset.get_loader(
            self.cfg.BATCH_SIZE, self.cfg.WORKERS
        )
        for e in tqdm(range(start_epoch, epochs + 1), total=epochs):
            logging.info(
                "\n%s Epoch %s",
                time.strftime("%m/%d/%Y %I:%M:%S %p", time.localtime()),
                e,
            )
            self.scheduler.step()
            self.net.train()
            steps_pbar = tqdm(
                train_loader, total=self.cfg.EPOCH_SIZE // self.cfg.BATCH_SIZE
            )
            for data in steps_pbar:
                features, labels = data
                self.optimizer.zero_grad()
                features = features.float().to(self.device)
                labels = labels.float().to(self.device)
                output = self.net(features)
                loss = CrossEntropy2d(output, labels, weight=loss_weights)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                iter_ += 1
                steps_pbar.set_postfix({"loss": loss.item()})
            train_loss.append(np.mean(losses[-1 * self.cfg.EPOCH_SIZE :]))
            loss, iou_, acc_, f1_ = self.test()
            test_loss.append(loss)
            accu.append(acc_)
            iou.append(iou_ * 100)
            f1.append(f1_ * 100)
        disir_ext = f"DISIR{self.cfg.N_CLASSES}classes" if self.cfg.DISIR else ""
        name =  "_".join([os.path.join(self.cfg.PATH_MODELS, self.net_name), os.path.basename(self.dataset), f"{disir_ext}.pt"])
        self.save_to_jit(name ,self.disir)

    def _infer_image(self, stride, data):
        """infer for one image"""
        with torch.no_grad():
            img = data
            pred = np.zeros(
                img.shape[1:] + (self.cfg.N_CLASSES,)
            ) 
            occur_pix = np.zeros(
                (*img.shape[1:], 1)
            ) 
        for coords in grouper(
                self.cfg.BATCH_SIZE,
                sliding_window(img, step=stride, window_size=self.cfg.WINDOW_SIZE),
            ):
            data_patches = [from_coord_to_patch(img, coords, self.device)]
            data_patches = torch.cat([*data_patches], dim=1).float()
            outs = self.net(data_patches).data.cpu().numpy()

            for out, (x, y, w, h) in zip(outs, coords):
                out = out.transpose((1, 2, 0))
                pred[x : x + w, y : y + h] += out
                occur_pix[x : x + w, y : y + h, :] += 1
        pred = pred / occur_pix
        return pred

    def test(self, sparsity=0, use_previous_state=False, id_class=None, initial_file=None):
        """Test the network on images.
        Args:
            sparsity (int, optional): Number of clicks generated per image.
            use_previous_state (bool, optional): Use the previous pred to generate the click. 
            Only used if sparsity > 0. 
            id_class ([type], optional): class id of the newly sampled click.
            Only used if sparsity > 0. 
        """
        logging.info(
            "%s INFO: Begin testing",
            time.strftime("%m/%d/%Y %I:%M:%S %p", time.localtime()),
        )
        self.net.eval()
        loss, acc, iou, f1 = (
            [],
            [],
            [],
            [],
        )  
        test_dataset = RGBDataset(
            self.dataset,
            self.cfg,
            self.disir,
            False,
            0,
            use_previous_state,
            id_class,
        )
        test_images = test_dataset.get_loader(1, self.cfg.TEST_WORKERS)
        stride = self.cfg.STRIDE
        modif_pxls = 0
        wrong_pxls = 0
        sparsifier = Sparsifier(False, self.cfg)
        confusion_matrixes = np.zeros((sparsity, self.cfg.N_CLASSES, self.cfg.N_CLASSES))
        lr = self.cfg.CL_LR 
        for iteration, (idx, data) in enumerate(tqdm(zip(test_dataset.test_ids, test_images), total=len(test_dataset.test_ids))):
            if self.cfg.DISCA and not self.cfg.CL_SEQUENTIAL_LEARNING:
                self.load_weights(initial_file, is_jit=True)
            file_name = os.path.basename(sorted(glob(os.path.join(self.dataset, "gts", '*')))[idx])
            logging.info("Filename: %s", file_name)
            data = [i.squeeze(0) for i in data]
            img = data[:-1][0]
            gt = data[-1].cpu().numpy()
            raw_annots = np.zeros((self.cfg.N_CLASSES, *img.shape[1:]))
            img = torch.cat([img, torch.from_numpy(raw_annots)]).to(self.device)
            continual_optimizer = optim.SGD(self.net.parameters(), lr)
            is_cl = self.cfg.DISCA
            for s in tqdm(range(sparsity), total=sparsity):
                if self.disir and s > 0:
                    if self.cfg.GUIDED_FILTER:
                        img_guide = img[:self.cfg.IN_CHANNELS].cpu().numpy().transpose((1, 2, 0))
                    else:
                        img_guide = None
                    raw_annots = sparsifier.simulate_disir(gt, img_guide, pred, np.sum(raw_annots, 0), s, file_name, id_class)
                    raw_annots = raw_annots.transpose((2, 0, 1))
                    annots = encode_annots(self.cfg, img_guide, raw_annots)
                    annots = np.stack(annots)
                    img[self.cfg.IN_CHANNELS:self.cfg.IN_CHANNELS+self.cfg.N_CLASSES] = torch.from_numpy(annots).to(self.device)
                pred_ = self._infer_image(stride, img)
                if s != 0 and is_cl:
                    freeze_bn(self.net)
                    steps = self.cfg.CL_STEPS
                    print("learn\n")
                    for iteration in range(steps):
                        self.update_weights(img, pred_, torch.from_numpy(raw_annots).to(bool), continual_optimizer)
                    pred_ = self._infer_image(stride, img)

                loss.append(
                    CrossEntropy2d(
                        torch.from_numpy(np.expand_dims(pred_.transpose((2, 0, 1)), axis=0)),
                        torch.from_numpy(np.expand_dims(gt, axis=0)),
                    ).item()
                )
                pred = np.argmax(pred_, axis=-1)
                ignore_indx = None
                metric_iou = IoU(pred, gt, self.cfg.N_CLASSES, all_iou=use_previous_state, ignore_indx=ignore_indx)
                metric_f1 = f1_score(pred, gt, self.cfg.N_CLASSES, all=use_previous_state, ignore_indx=ignore_indx)
                if use_previous_state:
                    metric_iou, all_iou = metric_iou
                    metric_f1, all_f1, weighted_f1 = metric_f1
                if s == sparsity and self.cfg.DISCA and self.cfg.CL_FULL_PRED and metric_iou > 0.86:
                    lr2 = self.cfg.CL_LR * 10 * np.round(metric_iou, 2)
                    logging.info(f"Leraning on full prediction with learning rate {lr}.")
                    continual_optimizer2 = optim.SGD(self.net.parameters(), lr2)
                    steps = 10
                    pred_target = torch.argmax(torch.from_numpy(pred_).to(self.device), dim=-1).unsqueeze(0)
                    for step in range(steps):
                        for coords in grouper(
                        1,
                        sliding_window(img, step=self.cfg.STRIDE, window_size=self.cfg.WINDOW_SIZE),
                        ):
                            features = img
                            labels = pred_target
                            continual_optimizer2.zero_grad()
                            data_patches = from_coord_to_patch(features, coords, self.device).float()
                            data_patches[:, self.cfg.IN_CHANNELS:] = 0
                            target = from_coord_to_patch(labels, coords, self.device).long()
                            target = target[0]
                            pred = self.net(data_patches)
                            loss_ = nn.CrossEntropyLoss()(pred, target)
                            loss_.backward()
                            continual_optimizer2.step()
                metric_acc = accuracy(pred, gt, ignore_indx=ignore_indx)
                acc.append(metric_acc)
                iou.append(metric_iou)
                f1.append(metric_f1)
                logging.info("IoU: %s", metric_iou)
                logging.info("F1: %s", metric_f1)
                if use_previous_state:
                    name = os.path.join("/tmp/preds", self.cfg.NET_NAME + file_name)
                    self.metrics.loc[file_name, f"{s}_acc"] = metric_acc
                    self.metrics.loc[file_name, f"{s}_IoU"] = metric_iou
                    self.metrics.loc[file_name, f"{s}_F1"] = metric_f1
                    self.metrics.loc[file_name, f"{s}_F1_weighted"] = weighted_f1
                    for c, i in enumerate(all_iou):
                        self.metrics.loc[file_name, f"{s}_IoU_class_{c}"] = i
                    for c, i in enumerate(all_f1):
                        self.metrics.loc[file_name, f"{s}_F1_class_{c}"] = i
                    if s > 0:
                        diff = np.sum(old_pred != pred)
                        modif_pxls += diff
                        wrongs = np.sum(np.bitwise_and(old_pred != pred, pred != gt))
                        self.metrics.loc[file_name, f"{s}_wrong_pxls"] = wrongs
                        self.metrics.loc[file_name, f"{s}_good_pxls"] = diff - wrongs
                        wrong_pxls += wrongs
                    old_pred = pred.copy()
                    dataset_name = os.path.basename(self.dataset)
                    csv_name = "{}_{}.csv".format(os.path.join(self.cfg.SAVE_FOLDER, self.cfg.NET_NAME), dataset_name)
                    self.metrics.to_csv(csv_name)
                    # cm
                    confusion = metrics.confusion_matrix(pred.flatten(), gt.flatten())
                    confusion_matrixes[s] += confusion
                    matrix_name = csv_name.replace(".csv", "_confusion_matrix.npy")
                    np.save(matrix_name, confusion_matrixes)

        #  Update logger
        if use_previous_state:
            logging.info(
                "Total modified pixels: %s", modif_pxls / len(test_dataset.test_ids)
            )
            logging.info(
                "Wrong modified pixels: %s", wrong_pxls / len(test_dataset.test_ids)
            )
        logging.info("Mean IoU : " + str(np.nanmean(iou)))
        logging.info("Mean accu : " + str(np.nanmean(acc)))
        logging.info("Mean F1 : " + str(np.nanmean(f1)))
        return np.mean(loss), np.nanmean(iou), np.mean(acc), np.mean(f1)

    def update_weights(self, img, initial_pred, annots, optimizer):
        sparse_target = torch.full([*annots.shape[-2:]], -1, dtype=torch.long, device=self.device)
        for i in range(self.cfg.N_CLASSES):
            sparse_target[annots[i]] = i
        sparse_target = sparse_target.unsqueeze(0)
        # compute distance map (optionnal)
        if self.cfg.CL_DIST_TRANS:
            dm_annots = distance_transform_edt(~(np.sum(annots.cpu().numpy(), axis=0, dtype=bool)), return_distances=True)
            dm_annots = torch.from_numpy(dm_annots).to(self.device).unsqueeze(0)
            dm_annots = (dm_annots - torch.min(dm_annots)) / (torch.max(dm_annots) - torch.min(dm_annots))
        data = img
        initial_pred = initial_pred.transpose((2, 0, 1))
        for coords in grouper(
                1,
                sliding_window(data, step=self.cfg.STRIDE, window_size=self.cfg.WINDOW_SIZE),
            ):
            optimizer.zero_grad()
            annots_patches = from_coord_to_patch(annots, coords, "cpu")
            if torch.sum(annots_patches) == 0:
                continue
            data_patches = from_coord_to_patch(data, coords, self.device).float()
            if np.random.random() < 0.2:
                data_patches[:, self.cfg.IN_CHANNELS:] = 0
            target = from_coord_to_patch(sparse_target, coords, self.device).long()
            target = target[0]
            pred = self.net(data_patches)
            
            ini_pred = from_coord_to_patch(initial_pred, coords, self.device)
            if self.cfg.CL_DIST_TRANS:
                dm_annots_ = from_coord_to_patch(dm_annots, coords, self.device)

            loss = nn.CrossEntropyLoss(ignore_index=-1)(pred, target)
            reg = nn.L1Loss(reduction="none")(pred, ini_pred) if self.cfg.CL_REG == "L1" else  nn.CrossEntropyLoss(reduction="none")(pred, torch.argmax(ini_pred, dim=1))
            if self.cfg.CL_DIST_TRANS:
                reg = reg * dm_annots_
            reg = torch.mean(reg)
            loss = loss + reg   * self.cfg.CL_LAMBDA 
            loss.backward()
            optimizer.step()

    def _load_init(self, pretrain_file):
        if pretrain_file:
            checkpoint = torch.load(pretrain_file)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.net.load_state_dict(checkpoint["state_dict"])
            train_loss = checkpoint["train_loss"]
            test_loss = checkpoint["test_loss"]
            start_epoch = checkpoint["epoch"]
            losses = checkpoint["losses"]
            accu = checkpoint["accu"]
            iou = checkpoint["iou"]
            f1 = checkpoint["f1"]
            logging.info(
                "Loaded checkpoint '{}' (epoch {})".format(
                    pretrain_file, checkpoint["epoch"]
                )
            )
        else:
            start_epoch = 1
            train_loss = []
            test_loss = []
            losses = []
            accu = []
            iou = []
            f1 = []
        return start_epoch, accu, iou, f1, train_loss, test_loss, losses
