import numpy as np


def convert_to_color(arr_2d, n_classes):
    """ Numeric labels to RGB-color encoding. """
    if n_classes == 6:
        palette = {0: (255, 255, 255),  # Impervious surfaces (white)
                   1: (0, 0, 255),  # Buildings (blue)
                   2: (0, 255, 255),  # Low vegetation (cyan)
                   3: (0, 255, 0),  # Trees (green)
                   4: (255, 255, 0),  # Cars (yellow)
                   5: (255, 0, 0),  # Clutter (red)
                   6: (0, 0, 0)}  # Undefined (black)

        arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
        for c, i in palette.items():
            m = arr_2d == c
            arr_3d[m] = i
    elif n_classes == 2:
        arr_3d = arr_2d * 255
    elif n_classes == 4:
        palette = {0: (255, 0, 0),  # Impervious surfaces (white)
                   1: (0, 255, 0),  # Buildings (blue)
                   2: (0, 0, 255),  # Low vegetation (cyan)
                   3: (0, 0, 0),  # Undefined (black)
                   }
        arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
        for c, i in palette.items():
            m = arr_2d == c
            arr_3d[m] = i
    return arr_3d
