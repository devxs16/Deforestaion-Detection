from scipy.ndimage import label
import numpy as np


def merge_changes(changes, image_width, image_height, patch_size=64):

    cols = image_width // patch_size
    rows = image_height // patch_size

    mask = np.zeros((rows, cols), dtype=np.uint8)

    for change in changes:

        col = change["x"] // patch_size
        row = change["y"] // patch_size

        if row < rows and col < cols:
            mask[row, col] = 1

    labeled, num_regions = label(mask)

    return labeled, num_regions