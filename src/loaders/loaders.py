"""
Pytorch loader
"""
import logging
import os
import warnings
from glob import glob
from typing import Any, Dict, List

import cv2 as cv
import numpy as np
import numpy.random as random
import rasterio
from albumentations import (Compose, HorizontalFlip, HueSaturationValue,
                            RandomBrightnessContrast, RGBShift,
                            ShiftScaleRotate, VerticalFlip)
from icecream import ic
from rasterio.windows import Window
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.loaders.SegDataLoader import SegDataLoader
from src.loaders.sparsifier import Sparsifier

rasterio_loader = logging.getLogger("rasterio")
rasterio_loader.setLevel(logging.ERROR)  # rasterio outputs warnings with some tiff files.


class RGBDataset(SegDataLoader):
    def __init__(
            self,
            dataset: str,
            cfg: Dict[str, Any],
            disir: bool,
            train: bool = True,
            test_sparsity: int = 0,
            use_previous_state: bool = False,
            id_class: int = None,
    ):
        super().__init__(dataset, cfg)
        # Sanity checks
        if not cfg.TRANSFORMATION and cfg.HEAVY_AUG:
            raise KeyError("Please activate TRANSFORMATION or desactivate HEAVY_AUG.")
        self.test_sparsity = test_sparsity
        self.disir = disir
        self.cfg = cfg
        self.train = train
        self.use_previous_state = use_previous_state
        self.id_class = id_class
        self.rgb = False
        self.augmentation = cfg.TRANSFORMATION
        self.col_jit = cfg.COL_JIT  # Color jittering
        self.heavy_aug = int(cfg.HEAVY_AUG)  # Rotation, etc...
        self.sparsifier = Sparsifier(train, cfg, None)
        assert cfg.IN_CHANNELS == 3
        train_ids, test_ids = self.split_dataset(cfg.test_size)
        self.train_ids = train_ids
        self.test_ids = test_ids
        gts = sorted(glob(os.path.join(dataset, "gts/*")))
        ext_gt = '.' + gts[0].split('.')[1]
        ext_imgs = '.' + sorted(glob(os.path.join(dataset, "imgs/*")))[0].split('.')[1]
        imgs = [file.replace("gts", "imgs").replace(ext_gt, ext_imgs) for file in gts]
        # the following are used to test Revolver
        annot_files = [os.path.join("/tmp/annots", cfg.NET_NAME + os.path.basename(file)) for file in gts]
        pred_files = [os.path.join("/tmp/preds", cfg.NET_NAME + os.path.basename(file)) for file in gts]
        if train:
            self.gts = [gts[i] for i in train_ids]
            self.imgs = [imgs[i] for i in train_ids]
        else:
            self.gts = [gts[i] for i in test_ids]
            self.imgs = [imgs[i] for i in test_ids]
        if use_previous_state:
            self.pred_files = [pred_files[i] for i in test_ids]
            self.annot_files = [annot_files[i] for i in test_ids]
            os.makedirs(os.path.dirname(self.pred_files[0]), exist_ok=True)
            os.makedirs(os.path.dirname(self.annot_files[0]), exist_ok=True)
            if not test_sparsity:
                # ie if it's the first pass, delete old files
                for (i, j) in zip(self.pred_files, self.annot_files):
                    if os.path.exists(i):
                        os.remove(i)
                    if os.path.exists(j):
                        os.remove(j)
        self.means, self.stds = self.mean_std()

    def _load_data(self, i):
        """Load data"""
        # Define the window and and the read image.
        if self.train:
            #  Pick a random image and randomly set the coordinates of the crop
            random_id = random.randint(len(self.gts))
            with rasterio.open(self.gts[random_id]) as src:
                x_crop, y_crop = (
                    random.randint(max(1, src.shape[1] - self.cfg.WINDOW_SIZE[1])),
                    random.randint(max(1, src.shape[0] - self.cfg.WINDOW_SIZE[0])),
                )
                window = Window(x_crop, y_crop, self.cfg.WINDOW_SIZE[1], self.cfg.WINDOW_SIZE[0])
                del (src, x_crop, y_crop)
        else:
            # Not random for test and load the full images
            random_id = i
            window = None

        with rasterio.open(self.imgs[random_id]) as src:
            img = np.asarray(1 / 255 * src.read(window=window), dtype=np.float32)[:3].transpose(
                (1, 2, 0)
            )
            img = ((img - self.means) / self.stds).astype(np.float32)
            features = img
        with rasterio.open(self.gts[random_id]) as src:
            labels = src.read(1, window=window)
        if self.cfg.N_CLASSES == 2 and "Potsdam" in self.gts[0]:
            # Implies building segmentation only on Potsdam and match AIRS resolution
            labels[labels != 1] = 0
            if labels.shape[0] == 6000:
                features = cv.resize(features, (4000, 4000))
                labels = cv.resize(labels, (4000, 4000))
        do_load = self.use_previous_state and self.test_sparsity
        if do_load:
            os.path.exists(self.annot_files[random_id])
        previous_pred = rasterio.open(self.pred_files[random_id]).read(1, window=window) if do_load else None
        previous_annot = rasterio.open(self.annot_files[random_id]).read(1, window=window) if do_load else None
        return features, labels, previous_pred, previous_annot

    def _data_augmentation(self, features, labels):
        """data augmentation"""
        transform = Compose(
            [HorizontalFlip(), VerticalFlip(), Compose([ShiftScaleRotate()], p=self.heavy_aug)], p=1
        )
        transform = transform(image=features, mask=labels)
        features = transform["image"]
        labels = transform["mask"]

        if self.col_jit:
            transform = Compose(
                [HueSaturationValue(p=0.3), RGBShift(p=0.2), RandomBrightnessContrast(p=0.5)], p=1
            )
            features = transform(image=features)["image"]
        return features, labels

    def __getitem__(self, i):
        """
        Sparsity and augmentation are applied if it was enabled in cfg.
        Returns
        -------
        Data and ground truth in the right tensor shape.

        """
        # load data
        with warnings.catch_warnings():
            # Remove warnings when image is not georeferenced.
            warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
            features, labels, previous_pred, previous_annot = self._load_data(i)

        #  Data augmentation
        if self.train and self.augmentation:
            features, labels = self._data_augmentation(features, labels)

        # disir
        if self.disir and self.train:
            if self.use_previous_state:
                name = self.annot_files[i]
                name = os.path.join(os.path.dirname(name), os.path.basename(self.gts[i]))

            else:
                name = None
            train_sparsity = random.randint(1, 101) if random.random() > 0.3 else 0
            sparsity = train_sparsity if self.train else self.test_sparsity
            if self.cfg.GUIDED_FILTER:
                img_guide = features[..., :self.cfg.IN_CHANNELS]
            else:
                img_guide = None
            annots = self.sparsifier.simulate_disir(
                labels, img_guide, previous_pred, previous_annot, sparsity, name, self.id_class
                )
            features = np.concatenate([features, annots], axis=2)

        features = features.transpose((2, 0, 1))
        return features, labels


    def mean_std(self):
        dataset = self.dataset
        if self.cfg.MEAN_STD:
            rgb = [[], [], []]
            for idx in tqdm(glob(dataset + "imgs/*"), total=len(glob(self.dataset + "imgs/*"))):
                with rasterio.open(idx) as src:
                    img = np.asarray(1 / 255 * src.read(), dtype="float32").transpose((1, 2, 0))
                    for i in range(3):
                        rgb[i].extend(img[:, :, i].flatten())
            means = [np.mean(i) for i in rgb]
            stds = [np.std(i) for i in rgb]
            logging.info(" RGB means/stds: %s, %s", means, stds)
            del img, rgb
        else:
            means = [0, 0, 0]
            stds = [1, 1, 1]
        return means, stds

    def set_sparsifier_weights(self, weights):
        self.sparsifier.weights = weights


class GTDataset(SegDataLoader):
    """Only load ground truth. Used to compute classes frequency."""
    def __init__(
            self,
            dataset: str,
            cfg: Dict[str, Any],
            ids: List[int]
    ):
        self.cfg = cfg
        self.dataset = dataset
        gts = sorted(glob(os.path.join(dataset, "gts/*")))
        self.gts = []
        self.gts = [gts[i] for i in ids]

    def _load_data(self, i):
        """Load data"""
        with rasterio.open(self.gts[i]) as src:
            labels = src.read(1)
        return labels

    def __len__(self):
        return len(self.gts)

    def __getitem__(self, i):
        """
        Sparsity and augmentation are applied if it was enabled in cfg.
        Returns
        -------
        Data and ground truth in the right tensor shape.

        """
        # load data
        with warnings.catch_warnings():
            # Remove warnings when image is not georeferenced.
            warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
            labels = self._load_data(i)
        return labels

    def split_dataset(self, test_size):
        # TODO: Refactoring neaded over here
        dataset_files = glob(os.path.join(self.dataset, "gts/*"))
        dataset_ids = np.arange(len(dataset_files))
        if test_size < 1:
            self.train_ids, self.test_ids = train_test_split(dataset_ids, test_size=test_size, random_state=42)
        else:
            self.train_ids = self.test_ids
        if len(self.train_ids) and len(self.test_ids):
            return self.train_ids, self.test_ids
        dataset_path = os.path.abspath(self.dataset)
        message = "Can't load dataset, propbably path is empty. \n {}".format(dataset_path)
        raise Exception(message)

    def compute_frequency(self):
        print("Computing weights...")
        weights = [[] for i in range(self.cfg.N_CLASSES)]
        labels = self.get_loader(1, 12)
        for gt in labels:
            for i in range(self.cfg.N_CLASSES):
                weights[i].append(np.where(gt == i)[0].shape[0])
        sum_pxls = np.sum(weights)
        weights = [1 / (np.sum(i) / sum_pxls) for i in weights]
        if self.cfg.N_CLASSES == 6:
            weights[-1] = min(weights)  # because clutter class is an ill-posed problem
        weights = np.asarray(weights)
        logging.info(f"Following weights have been computed: {weights}")
        ic(weights)
        return weights

    def _data_augmentation(self):
        pass

class NoAnnotsRGBDataset(RGBDataset):
    def __init__(
            self,
            dataset: str,
            cfg: Dict[str, Any],
            train: bool = True,
            annots_channels: int = 0
    ):
        super(NoAnnotsRGBDataset, self).__init__(dataset, cfg, disir=False, train=train)
        self.annots_channels = annots_channels

    def _load_data(self, i):
        """Load data"""
        # Define the window and and the read image.
            #  Pick a random image and randomly set the coordinates of the crop
        random_id = random.randint(len(self.gts))
        with rasterio.open(self.gts[random_id]) as src:
            x_crop, y_crop = (
                random.randint(max(1, src.shape[1] - self.cfg.WINDOW_SIZE[1])),
                random.randint(max(1, src.shape[0] - self.cfg.WINDOW_SIZE[0])),
            )
            window = Window(x_crop, y_crop, self.cfg.WINDOW_SIZE[1], self.cfg.WINDOW_SIZE[0])
            del (src, x_crop, y_crop)

        with rasterio.open(self.imgs[random_id]) as src:
            img = np.asarray(1 / 255 * src.read(window=window), dtype=np.float32)[:3].transpose(
                (1, 2, 0)
            )
            img = ((img - self.means) / self.stds).astype(np.float32)
            features = img
        with rasterio.open(self.gts[random_id]) as src:
            labels = src.read(1, window=window)
        if self.cfg.N_CLASSES == 2 and "Potsdam" in self.gts[0]:
            # Implies building segmentation only on Potsdam and match AIRS resolution
            labels[labels != 1] = 0
        return features, labels

    def __getitem__(self, i):
        """
        Sparsity and augmentation are applied if it was enabled in cfg.
        Returns
        -------
        Data and ground truth in the right tensor shape.

        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
            features, labels = self._load_data(i)
        features = features.transpose((2, 0, 1))
        if self.annots_channels > 0:
            features = np.cat([features, np.zeros((self.annots_channels, *labels.shape))])
        return [features, labels]

    def __len__(self):
        """Defines the length of an epoch"""
        return self.cfg.EPOCH_SIZE

