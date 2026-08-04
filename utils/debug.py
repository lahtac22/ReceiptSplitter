from pathlib import Path
import cv2

from config import DEBUG_DIR


def get_debug_folder(file_stem):
    """
    Create and return a debug folder for one input file.

    Example:
        debug/receipt1/
    """

    folder = DEBUG_DIR / file_stem

    folder.mkdir(parents=True, exist_ok=True)

    return folder


def save_debug_image(folder, image, filename):
    """
    Save an image into the file's debug folder.
    """

    output_file = folder / filename

    cv2.imwrite(str(output_file), image)

    print(f"[INFO] Saved {output_file}")