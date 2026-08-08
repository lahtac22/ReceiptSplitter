import time
import cv2
import numpy as np

from utils.files import get_input_files
from image.image_loader import load_image

from utils.debug import get_debug_folder
from utils.debug import save_debug_image

from utils.report import write_report

from detectors.colour_detector import detect
from detectors.split_detector import find_vertical_gaps

from detectors.paper_boundary_detector import (
    find_paper_regions,
    find_internal_boundaries,
    draw_paper_regions,
    draw_internal_boundaries
)

from utils.drawing import (
    draw_rectangles,
    draw_vertical_gaps,
    draw_contours
)


def main():

    files = get_input_files()

    print(
        f"\nFound {len(files)} file(s).\n"
    )

    for file in files:

        total_start = time.perf_counter()

        timings = {}

        print(
            f"[INFO] Loading {file.name}"
        )

        # ----------------------------------------------------
        # Create inspection folder
        # ----------------------------------------------------

        debug_folder = get_debug_folder(
            file.stem
        )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        start = time.perf_counter()

        image = load_image(
            file
        )

        timings["Load Image"] = (
            time.perf_counter()
            - start
        )

        print(
            f"[INFO] Image size: "
            f"{image.shape}"
        )

        save_debug_image(
            debug_folder,
            image,
            "01_loaded.png"
        )

        # ----------------------------------------------------
        # Existing receipt detection
        # ----------------------------------------------------

        start = time.perf_counter()

        (
            rectangles,
            mask,
            contours,
            stats
        ) = detect(
            image
        )

        timings["Receipt Detection"] = (
            time.perf_counter()
            - start
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

        print(
            f"[INFO] Found "
            f"{stats['contours_found']} contour(s)"
        )

        print(
            f"[INFO] Accepted "
            f"{stats['accepted_contours']} contour(s)"
        )

        print(
            f"[INFO] Found "
            f"{len(rectangles)} candidate receipt(s)"
        )

        # ----------------------------------------------------
        # Gap detection
        # ----------------------------------------------------

        start = time.perf_counter()

        gaps = find_vertical_gaps(
            mask
        )

        timings["Gap Detection"] = (
            time.perf_counter()
            - start
        )

        print(
            f"[INFO] Found "
            f"{len(gaps)} vertical gap(s)"
        )

        # ----------------------------------------------------
        # Draw existing detections
        # ----------------------------------------------------

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

        # ====================================================
        # PAPER-BOUNDARY ANALYSIS
        # ====================================================

        print(
            "[INFO] Running paper-boundary analysis..."
        )

        # ----------------------------------------------------
        # Find likely paper regions
        # ----------------------------------------------------

        start = time.perf_counter()

        (
            paper_regions,
            paper_mask,
            cleaned_paper_mask,
            paper_diagnostics
        ) = find_paper_regions(
            image
        )

        timings["Paper Region Detection"] = (
            time.perf_counter()
            - start
        )

        # ----------------------------------------------------
        # Save paper masks
        # ----------------------------------------------------

        save_debug_image(
            debug_folder,
            paper_mask,
            "06_paper_mask.png"
        )

        save_debug_image(
            debug_folder,
            cleaned_paper_mask,
            "06b_paper_cleaned.png"
        )

        print(
            f"[INFO] Paper regions found: "
            f"{paper_diagnostics['regions_found']}"
        )

        print(
            f"[INFO] Paper pixels: "
            f"{paper_diagnostics['paper_pixels']}"
        )

        # ----------------------------------------------------
        # Draw paper regions
        # ----------------------------------------------------

        paper_preview = draw_paper_regions(
            image,
            paper_regions
        )

        save_debug_image(
            debug_folder,
            paper_preview,
            "07_paper_regions.png"
        )

        # ----------------------------------------------------
        # Find possible internal paper boundaries
        # ----------------------------------------------------

        all_boundaries = []

        start = time.perf_counter()

        for region in paper_regions:

            (
                boundaries,
                boundary_diagnostics
            ) = find_internal_boundaries(
                image,
                region
            )

            all_boundaries.extend(
                boundaries
            )

        timings["Paper Boundary Detection"] = (
            time.perf_counter()
            - start
        )

        print(
            f"[INFO] Paper boundaries found: "
            f"{len(all_boundaries)}"
        )

        # ----------------------------------------------------
        # Print boundary information
        # ----------------------------------------------------

        for index, boundary in enumerate(
            all_boundaries,
            start=1
        ):

            print(
                f"       Boundary {index}: "
                f"{boundary['orientation']}, "
                f"position={boundary['position']}, "
                f"confidence="
                f"{boundary['confidence']:.3f}"
            )

        # ----------------------------------------------------
        # Draw paper boundaries
        # ----------------------------------------------------

        boundary_preview = (
            draw_internal_boundaries(
                image,
                all_boundaries
            )
        )

        save_debug_image(
            debug_folder,
            boundary_preview,
            "08_paper_boundaries.png"
        )

        # ====================================================
        # SPLIT PAPER REGIONS
        # ====================================================

        print(
            "[INFO] Splitting paper regions..."
        )

        # ----------------------------------------------------
        # Separate vertical and horizontal boundaries
        # ----------------------------------------------------

        vertical_boundaries = [
            boundary
            for boundary in all_boundaries
            if boundary["orientation"] == "vertical"
        ]

        horizontal_boundaries = [
            boundary
            for boundary in all_boundaries
            if boundary["orientation"] == "horizontal"
        ]

        print(
            f"[INFO] Splitter boundaries: "
            f"{len(vertical_boundaries)} vertical, "
            f"{len(horizontal_boundaries)} horizontal"
        )

        receipts = []

        image_height, image_width = image.shape[:2]

        # ----------------------------------------------------
        # Build vertical split positions
        # ----------------------------------------------------

        vertical_positions = [0]

        for boundary in vertical_boundaries:

            position = int(
                boundary["position"]
            )

            if (
                position > 0
                and position < image_width
            ):
                vertical_positions.append(
                    position
                )

        vertical_positions.append(
            image_width
        )

        vertical_positions = sorted(
            set(vertical_positions)
        )

        # ----------------------------------------------------
        # Build horizontal split positions
        # ----------------------------------------------------

        horizontal_positions = [0]

        for boundary in horizontal_boundaries:

            position = int(
                boundary["position"]
            )

            if (
                position > 0
                and position < image_height
            ):
                horizontal_positions.append(
                    position
                )

        horizontal_positions.append(
            image_height
        )

        horizontal_positions = sorted(
            set(horizontal_positions)
        )

        # ----------------------------------------------------
        # No usable boundaries
        # ----------------------------------------------------

        if (
            len(vertical_positions) == 2
            and len(horizontal_positions) == 2
        ):

            print(
                "[INFO] No usable boundaries. "
                "Using paper-mask splitting."
            )

            for region in paper_regions:

                x = int(
                    region["x"]
                )

                y = int(
                    region["y"]
                )

                w = int(
                    region["width"]
                )

                h = int(
                    region["height"]
                )

                crop = image[
                    y:y + h,
                    x:x + w
                ].copy()

                if crop.size == 0:
                    continue

                receipts.append(
                    {
                        "image": crop,
                        "left": x,
                        "right": x + w,
                        "top": y,
                        "bottom": y + h,
                        "method": "paper-mask"
                    }
                )

        # ----------------------------------------------------
        # Use detected boundaries as a grid
        # ----------------------------------------------------

        else:

            print(
                f"[INFO] Split grid: "
                f"{len(vertical_positions) - 1} columns × "
                f"{len(horizontal_positions) - 1} rows"
            )

            # ------------------------------------------------
            # Create grid cells
            # ------------------------------------------------

            for row in range(
                len(horizontal_positions) - 1
            ):

                top = horizontal_positions[
                    row
                ]

                bottom = horizontal_positions[
                    row + 1
                ]

                for column in range(
                    len(vertical_positions) - 1
                ):

                    left = vertical_positions[
                        column
                    ]

                    right = vertical_positions[
                        column + 1
                    ]

                    if right <= left:
                        continue

                    if bottom <= top:
                        continue

                    cell = image[
                        top:bottom,
                        left:right
                    ]

                    if cell.size == 0:
                        continue

                    # ----------------------------------------
                    # Calculate paper fraction
                    # ----------------------------------------

                    hsv = cv2.cvtColor(
                        cell,
                        cv2.COLOR_BGR2HSV
                    )

                    saturation = hsv[:, :, 1]
                    value = hsv[:, :, 2]

                    paper_mask_cell = (
                        (value >= 125)
                        &
                        (saturation <= 100)
                    )

                    paper_fraction = (
                        np.count_nonzero(
                            paper_mask_cell
                        )
                        / paper_mask_cell.size
                    )

                    # ----------------------------------------
                    # Reject cells containing very little
                    # paper.
                    # ----------------------------------------

                    if paper_fraction < 0.18:
                        continue

                    receipts.append(
                        {
                            "image": cell.copy(),
                            "left": left,
                            "right": right,
                            "top": top,
                            "bottom": bottom,
                            "method": "boundary-grid",
                            "paper_fraction":
                                paper_fraction
                        }
                    )

                    print(
                        f"[INFO] Accepted grid cell: "
                        f"x={left}–{right}, "
                        f"y={top}–{bottom}, "
                        f"paper={paper_fraction:.3f}"
                    )

        # ----------------------------------------------------
        # Sort receipts in reading order
        # ----------------------------------------------------

        receipts.sort(
            key=lambda receipt: (
                receipt["top"],
                receipt["left"]
            )
        )

        print(
            f"[INFO] Created "
            f"{len(receipts)} receipt(s)"
        )

        # ----------------------------------------------------
        # Save receipt images
        # ----------------------------------------------------

        for index, receipt in enumerate(
            receipts,
            start=1
        ):

            filename = (
                f"09_receipt_{index:02d}.png"
            )

            save_debug_image(
                debug_folder,
                receipt["image"],
                filename
            )

            print(
                f"       Receipt {index}: "
                f"x={receipt['left']}–"
                f"{receipt['right']}, "
                f"y={receipt['top']}–"
                f"{receipt['bottom']} "
                f"({receipt['method']})"
            )

        # ----------------------------------------------------
        # Total time
        # ----------------------------------------------------

        timings["Total"] = (
            time.perf_counter()
            - total_start
        )

        # ----------------------------------------------------
        # Existing report
        # ----------------------------------------------------

        write_report(
            debug_folder,
            file.name,
            image,
            rectangles,
            gaps,
            timings,
            stats
        )

        print(
            "[INFO] Inspection report written."
        )

        print(
            f"[INFO] Total time: "
            f"{timings['Total']:.3f} s"
        )

        print()


if __name__ == "__main__":
    main()