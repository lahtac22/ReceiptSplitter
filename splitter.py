from utils.files import get_input_files
from image.image_loader import load_image
from utils.debug import get_debug_folder
from utils.debug import save_debug_image

from detectors.colour_detector import detect
from detectors.split_detector import find_vertical_gaps

from utils.drawing import draw_rectangles
from utils.drawing import draw_vertical_gaps


def main():

    files = get_input_files()

    print(f"\nFound {len(files)} file(s).\n")

    for file in files:

        print(f"[INFO] Loading {file.name}")

        # Create a debug folder for this file
        debug_folder = get_debug_folder(file.stem)

        # Load image
        image = load_image(file)

        print(f"[INFO] Image size: {image.shape}")

        # Save original image
        save_debug_image(
            debug_folder,
            image,
            "01_loaded.png"
        )

        # Detect paper
        rectangles, mask = detect(image)

        # Save mask
        save_debug_image(
            debug_folder,
            mask,
            "03_mask.png"
        )

        print(f"[INFO] Found {len(rectangles)} candidate receipt(s)")

        # Find vertical gaps
        gaps = find_vertical_gaps(mask)

        print(f"[INFO] Found {len(gaps)} vertical gap(s)")

        # Draw results
        preview = draw_rectangles(image, rectangles)
        preview = draw_vertical_gaps(preview, gaps)

        # Save detection preview
        save_debug_image(
            debug_folder,
            preview,
            "05_detected.png"
        )

        print()


if __name__ == "__main__":
    main()