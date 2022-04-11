import logging
from time import time

import buzzard as buzz
import numpy as np
from numpy import random
from src.utils.image import encode_annots

shapely_loader = logging.getLogger("shapely")
shapely_loader.setLevel(logging.ERROR)


class Sparsifier:
    def __init__(self, train, cfg, weights=None):
        self.train = train
        self.weights = weights
        self.n_classes = cfg.N_CLASSES
        self.cfg = cfg

    def simulate_disir(self, labels, img_guide, previous_pred, previous_annot, sparsity, name, class_id):
        diff_gt_pred = (labels != previous_pred) if previous_pred is not None else None
        if diff_gt_pred is not None and class_id is not None:
            diff_gt_pred[labels != class_id] = 0
        
        sparse_gt = self._simulate_sparsity(labels, previous_annot, diff_gt_pred, sparsity)
        if name:
            if previous_pred is not None:
                new_annot = np.where(previous_annot != sparse_gt)
                if len(new_annot[0]) < 1:
                    logging.info(f"No new annots added.")

        annots = [sparse_gt * (labels == i) for i in range(self.n_classes)]
        if self.train:
            annots = encode_annots(self.cfg, img_guide, annots)
        annots = np.array(annots)
        annots = annots.transpose((1, 2, 0))
        return annots

    def _simulate_sparsity(self, gt, old_annot, diff_gt_pred, sparsity):
        """
        3 cases:
        - Training (simulate at random)
        - Test initialization or train without annotations(no annotations)
        - Test after at least one inference (simulate using the wrong prediction map)
        """
        if sparsity == 0:
            sparse_gt = np.zeros_like(gt)
        else:
            if self.train:
                flat_gt = gt.reshape((-1))
                tot_pixs = flat_gt.shape[0]
                probs = np.ones((tot_pixs), dtype=np.float32)
                if self.weights is not None:
                    for i in range(self.n_classes):
                        probs[flat_gt == i] = self.weights[i]
                probs /= np.sum(probs)  # normalize
                sparse_points = random.choice(np.prod(gt.shape), sparsity, replace=False, p=probs)

                sparse_gt = np.zeros_like(flat_gt)
                sparse_gt[sparse_points] = 1
                sparse_gt = sparse_gt.reshape(*gt.shape)
            else:
                sparse_gt = old_annot.copy()
                fp = buzz.Footprint(gt=(0, .1, 0, 10, 0, -.1), rsize=gt.swapaxes(1, 0).shape)
                polygons = fp.find_polygons(diff_gt_pred)
                if not len(polygons):
                    return sparse_gt
                order = np.argsort([p.area for p in polygons])[max(np.random.randint(-8, 0), -len(polygons))]
                polygon = polygons[order]
                center = fp.spatial_to_raster(np.asarray(polygon.representative_point().xy).transpose((1, 0)))[0]
                loc = [
                    min(sparse_gt.shape[0] - 1, max(0, center[1]+0)),
                    min(sparse_gt.shape[1] - 1, max(0, center[0]+0)),
                        ]

                sparse_gt[loc[0], loc[1]] = 1
        return sparse_gt
