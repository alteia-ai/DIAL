import colorsys
import itertools
import random

import buzzard as buzz
import cv2 as cv
import numpy as np
import torch
from cv2.ximgproc import guidedFilter
from PIL import Image, ImageFilter
from skimage import measure


def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[1], step):
        if x + window_size[0] > top.shape[1]:
            x = top.shape[1] - window_size[0]
        for y in range(0, top.shape[2], step):
            if y + window_size[1] > top.shape[2]:
                y = top.shape[2] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def random_colors(n, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / n, 1, brightness) for i in range(n)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    np.random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, i=1, alpha=0.5):
    """Apply the given mask(==i) to the image. Binary mask.
    """
    target = image.copy()
    for c in range(3):
        target[:, :, c] = np.where(
            mask == i, image[:, :, c] * (1 - alpha) + alpha * color[c], image[:, :, c]
        )
    return target


def from_coord_to_patch(img, coords, device):
    """Returns patches of the input image. coors is an output of grouper(n, sliding window(...))"""
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    image_patches = [img[:, x : x + w, y : y + h] for x, y, w, h in coords]
    image_patches = torch.stack(image_patches).to(device)
    return image_patches


def encode_annots(cfg, img_guide, raw_annots):
    """img_guide: img if guided_filter. None else.
    """
    if cfg.DISTANCE_TRANSFORM:
        size = tuple(cfg.ENCODING_SIZE)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, size)
    else:
        kernel = np.ones((3, 3), dtype=np.uint8)
    encoded_gt = [cv.dilate(i.astype(np.uint8), kernel)  for i in raw_annots]
    if cfg.DISTANCE_TRANSFORM:
        encoded_gt = [cv.distanceTransform(i.astype(np.uint8), cv.DIST_L2, 3) for i in encoded_gt]
        if cfg.GUIDED_FILTER:
            r = 4
            eps = 0.005
            img_guide = img_guide.astype(np.float32)
            encoded_gt = [guidedFilter(img_guide, i, r, eps) for i in encoded_gt]
    return encoded_gt
