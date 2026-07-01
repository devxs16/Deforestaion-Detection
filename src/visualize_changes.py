import cv2
import numpy as np
from PIL import Image

from src.postprocess import merge_changes


def generate_change_map(image_path, changes, output_path="change_map.png"):

    image = Image.open(image_path)
    image_np = np.array(image)

    width, height = image.size

    labeled, num_regions = merge_changes(
        changes,
        width,
        height
    )

    PATCH_SIZE = 64

    for region in range(1, num_regions + 1):

        rows, cols = np.where(labeled == region)

        if len(rows) == 0:
            continue

        region_mask = np.zeros((height, width), dtype=np.uint8)

        for row, col in zip(rows, cols):

            x = col * PATCH_SIZE
            y = row * PATCH_SIZE

            region_mask[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ] = 255

        contours, _ = cv2.findContours(
            region_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            image_np,
            contours,
            -1,
            (255, 0, 0),
            3
        )

    Image.fromarray(image_np).save(output_path)

    return output_path


if __name__ == "__main__":

    from src.change_detection import detect_changes

    changes = detect_changes(
        "2018.jpg",
        "2026.jpg"
    )

    output = generate_change_map(
        "2026.jpg",
        changes
    )

    print(f"Saved to {output}")