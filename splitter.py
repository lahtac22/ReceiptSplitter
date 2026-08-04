from utils.files import get_input_files
from image.image_loader import load_image
from utils.debug import save_debug_image

from detectors.colour_detector import detect
from detectors.split_detector import find_vertical_gaps

from utils.drawing import draw_rectangles
from utils.drawing import draw_vertical_gaps


def main():

    files = get_input_files()

    print(f"\nFound {len(files)} file(s).\n")

    for file in files:

        print(f"Loading {file.name}...")

        image = load_image(file)

        print(f"Image size: {image.shape}")

        loaded_name = file.stem + "_loaded.png"
        mask_name = file.stem + "_mask.png"
        detected_name = file.stem + "_detected.png"

        save_debug_image(image, loaded_name)

        rectangles, mask = detect(image)

        save_debug_image(mask, mask_name)

        print(f"Found {len(rectangles)} candidate receipt(s)")

        gaps = find_vertical_gaps(mask)

        print(f"Found {len(gaps)} vertical gap(s)")

        preview = draw_rectangles(image, rectangles)

        preview = draw_vertical_gaps(preview, gaps)

        save_debug_image(preview, detected_name)

        print()


if __name__ == "__main__":
    main()