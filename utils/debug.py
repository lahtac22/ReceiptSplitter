import cv2
from pathlib import Path

from config import DEBUG_DIR


def save_debug_image(image, filename):
    """
    Save an image to the debug folder.
    """

    DEBUG_DIR.mkdir(exist_ok=True)

    output_file = DEBUG_DIR / filename

    cv2.imwrite(str(output_file), image)

    print(f"Saved {output_file.name}")