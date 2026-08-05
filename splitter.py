import time

from utils.files import get_input_files
from image.image_loader import load_image

from utils.debug import get_debug_folder
from utils.debug import save_debug_image

from utils.report import write_report

from detectors.colour_detector import detect
from detectors.split_detector import find_vertical_gaps

from utils.drawing import (
    draw_rectangles,
    draw_vertical_gaps,
    draw_contours
)


def main():

    files = get_input_files()

    print(f"\nFound {len(files)} file(s).\n")

    for file in files:

        total_start = time.perf_counter()

        timings = {}

        print(f"[INFO] Loading {file.name}")

        # Create inspection folder
        debug_folder = get_debug_folder(file.stem)

        # ----------------------------
        # Load image
        # ----------------------------
        start = time.perf_counter()

        image = load_image(file)

        timings["Load Image"] = time.perf_counter() - start

        print(f"[INFO] Image size: {image.shape}")

        save_debug_image(
            debug_folder,
            image,
            "01_loaded.png"
        )

        # ----------------------------
        # Detect receipts
        # ----------------------------
        start = time.perf_counter()

        rectangles, mask, contours, stats = detect(image)

        timings["Receipt Detection"] = (
            time.perf_counter() - start
        )

        save_debug_image(
            debug_folder,
            mask,
            "03_mask.png"
        )

        contour_image = draw_contours(
            image,
            contours
        )

        save_debug_image(
            debug_folder,
            contour_image,
            "04_contours.png"
        )

        print(f"[INFO] Found {stats['contours_found']} contour(s)")
        print(f"[INFO] Accepted {stats['accepted_contours']} contour(s)")
        print(f"[INFO] Found {len(rectangles)} candidate receipt(s)")

        # ----------------------------
        # Gap detection
        # ----------------------------
        start = time.perf_counter()

        gaps = find_vertical_gaps(mask)

        timings["Gap Detection"] = (
            time.perf_counter() - start
        )

        print(f"[INFO] Found {len(gaps)} vertical gap(s)")

        # ----------------------------
        # Draw detections
        # ----------------------------
        preview = draw_rectangles(
            image,
            rectangles
        )

        preview = draw_vertical_gaps(
            preview,
            gaps
        )

        save_debug_image(
            debug_folder,
            preview,
            "05_detected.png"
        )

        # ----------------------------
        # Total time
        # ----------------------------
        timings["Total"] = (
            time.perf_counter() - total_start
        )

        # ----------------------------
        # Report
        # ----------------------------
        write_report(
            debug_folder,
            file.name,
            image,
            rectangles,
            gaps,
            timings
        )

        print("[INFO] Inspection report written.")

        print(f"[INFO] Total time: {timings['Total']:.3f} s")

        print()


if __name__ == "__main__":
    main()