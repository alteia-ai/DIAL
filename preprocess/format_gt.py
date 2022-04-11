import argparse
from glob import glob

import click
import cv2
import numpy as np
import rasterio as rio
from PIL import Image
from tqdm import tqdm


@click.command()
@click.option('-d', "--directory", help="Ground-truth directory", type=str, required=True)
@click.option('-c', "--n_classes", help="Number of classes.", type=int, required=True)
@click.option("--safety/--no-safety", help="Activate or desactivate safety alert.", default=True)
def format_gt(directory, n_classes, safety):
    if safety:
        k = input(
            "This will modify the ground truth maps in your folder. If you don't have a copy of your current ground\
             truth maps, you're advised to do so.If you want to disable this message, add '-s' to your cli. Type\
              'y' to continue, any other key to stop.\n")
        if k != 'y':
            raise Exception("It was another key")
    files = glob(directory + '*')
    for file in tqdm(files, total=len(files)):
        reformat_gt(file, n_classes)
    print("Conversion done.")


def convert_from_color(arr_3d, n_classes):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    if n_classes == 6:
        palette = {0: (255, 255, 255),  # Impervious surfaces (white)
                   1: (0, 0, 255),  # Buildings (blue)
                   2: (0, 255, 255),  # Low vegetation (cyan)
                   3: (0, 255, 0),  # Trees (green)
                   4: (255, 255, 0),  # Cars (yellow)
                   5: (255, 0, 0),  # Clutter (red)
                   6: (0, 0, 0)}  # Undefined (black)

        invert_palette = {v: k for k, v in palette.items()}
        for c, i in invert_palette.items():
            m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
            arr_2d[m] = i
    elif n_classes == 7: # SemCity Toulouse
        palette = {
            0: (255, 255, 255), # Void
            1: (38, 38, 38), # Impervious surface
            2: (238, 118, 33), # Building
            3: (34, 139, 34), # Pervious surface
            4: (000, 222, 137), # High vegetation
            5: (255, 000, 000), # Car
            6: (000, 000, 238), # Water
            7: (160, 30, 230), # Sport venues
            }
        invert_palette = {v: k for k, v in palette.items()}
        for c, i in invert_palette.items():
            m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
            arr_2d[m] = i
    return arr_2d


def reformat_gt(gt, n_classes):
    with rio.open(gt) as src:
        meta = src.meta
        meta.update(dtype=np.uint8, compress="LZW", count=1)
        img = convert_from_color(src.read()[:3].transpose((1, 2, 0)), n_classes)
        height = src.height
        width = src.width
    if meta['crs'] is not None:
        with rio.open(gt, "w", **meta) as out_file:
            out_file.write(img.astype(np.uint8), indexes=1)
    else:
        with rio.open(gt, "w", dtype=np.uint8, compress="LZW", count=1, driver="GTiff", height=height,
                      width=width) as out_file:
            out_file.write(img.astype(np.uint8), indexes=1)
    return

if __name__ == "__main__":
    format_gt()
