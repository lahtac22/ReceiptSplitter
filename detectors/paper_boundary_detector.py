import cv2
import numpy as np


# ============================================================
# Paper Boundary Detector
# ============================================================
# Diagnostic detector for finding physical paper regions in
# receipt photographs.
#
# It does not use OCR or Hough rectangles.
#
# The internal-boundary detector looks for long, continuous
# physical edges rather than relying on a difference in
# paper/background colour.
# ============================================================


def _paper_likelihood_mask(image):
    """Create a binary mask for pixels likely to be paper."""

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    _, saturation, value = cv2.split(hsv)

    bright = value >= 125
    low_saturation = saturation <= 100

    mask = np.zeros(
        image.shape[:2],
        dtype=np.uint8
    )

    mask[bright & low_saturation] = 255

    return mask


def _clean_paper_mask(mask):
    """Close small holes and remove isolated small regions."""

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (21, 21)
    )

    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel
    )

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        open_kernel
    )

    return cleaned


def _contour_rectangle(contour):
    x, y, w, h = cv2.boundingRect(contour)

    return (
        int(x),
        int(y),
        int(w),
        int(h)
    )


def _contour_score(contour, image_shape):
    """Score a paper-like connected region."""

    image_height, image_width = image_shape[:2]

    area = cv2.contourArea(contour)

    x, y, w, h = cv2.boundingRect(contour)

    if w <= 0 or h <= 0:
        return 0.0

    rectangularity = area / (w * h)

    image_area = image_width * image_height

    area_ratio = (
        area / image_area
        if image_area
        else 0.0
    )

    area_component = min(
        1.0,
        area_ratio * 4.0
    )

    return float(
        0.65 * area_component
        + 0.35 * rectangularity
    )


def find_paper_regions(
    image,
    min_area_ratio=0.005,
    max_regions=20
):
    """
    Find likely paper regions.

    Returns:
        regions,
        raw_mask,
        cleaned_mask,
        diagnostics
    """

    if image is None or image.size == 0:

        return (
            [],
            None,
            None,
            {
                "regions_found": 0,
                "paper_pixels": 0
            }
        )

    raw_mask = _paper_likelihood_mask(
        image
    )

    cleaned_mask = _clean_paper_mask(
        raw_mask
    )

    contours, _ = cv2.findContours(
        cleaned_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = (
        image.shape[0]
        * image.shape[1]
    )

    minimum_area = (
        image_area
        * min_area_ratio
    )

    regions = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < minimum_area:
            continue

        x, y, w, h = _contour_rectangle(
            contour
        )

        regions.append(
            {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "area": float(area),
                "score": _contour_score(
                    contour,
                    image.shape
                )
            }
        )

    regions.sort(
        key=lambda item: (
            item["score"],
            item["area"]
        ),
        reverse=True
    )

    regions = regions[:max_regions]

    diagnostics = {
        "regions_found": len(regions),
        "paper_pixels": int(
            np.count_nonzero(
                cleaned_mask
            )
        )
    }

    return (
        regions,
        raw_mask,
        cleaned_mask,
        diagnostics
    )


def _smooth_projection(
    values,
    kernel_size
):
    """Smooth a one-dimensional projection."""

    if len(values) == 0:
        return values

    kernel_size = max(
        3,
        int(kernel_size)
    )

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = (
        np.ones(
            kernel_size,
            dtype=np.float32
        )
        / kernel_size
    )

    return np.convolve(
        values,
        kernel,
        mode="same"
    )


def _find_runs(boolean_array):
    """Return contiguous True runs as (start, end)."""

    runs = []

    in_run = False
    start = 0

    for index, value in enumerate(
        boolean_array
    ):

        if value and not in_run:

            start = index
            in_run = True

        elif not value and in_run:

            runs.append(
                (
                    start,
                    index
                )
            )

            in_run = False

    if in_run:

        runs.append(
            (
                start,
                len(boolean_array)
            )
        )

    return runs


def _longest_run_length(runs):

    if not runs:
        return 0

    return max(
        end - start
        for start, end in runs
    )


# ============================================================
# INTERNAL PAPER BOUNDARIES
# ============================================================

def find_internal_boundaries(
    image,
    region,
    min_boundary_length_ratio=0.30,
    min_boundary_width=3
):
    """
    Find long physical paper boundaries inside a paper region.

    Unlike the previous version, this does not depend on a loss
    of paper-likelihood.

    This is important when the background and paper have similar
    brightness/colour.

    Instead, it looks for a strong, continuous image edge running
    approximately vertically or horizontally.

    Returns:
        boundaries,
        diagnostics
    """

    x = int(region["x"])
    y = int(region["y"])
    w = int(region["width"])
    h = int(region["height"])

    if w <= 0 or h <= 0:

        return (
            [],
            {
                "vertical_candidates": 0,
                "horizontal_candidates": 0
            }
        )

    candidate = image[
        y:y + h,
        x:x + w
    ]

    if candidate.size == 0:

        return (
            [],
            {
                "vertical_candidates": 0,
                "horizontal_candidates": 0
            }
        )

    # --------------------------------------------------------
    # Convert to grayscale.
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        candidate,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Reduce small details such as text and tiny wrinkles.
    # --------------------------------------------------------

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    boundaries = []

    # ========================================================
    # VERTICAL EDGES
    # ========================================================

    gradient_x = np.abs(
        cv2.Sobel(
            gray,
            cv2.CV_32F,
            1,
            0,
            ksize=3
        )
    )

    gradient_threshold = max(
        20.0,
        float(
            np.percentile(
                gradient_x,
                90
            )
        )
    )

    edge_pixels = (
        gradient_x
        >= gradient_threshold
    )

    # How much of each column contains
    # a strong vertical edge?
    vertical_strength = np.mean(
        edge_pixels,
        axis=0
    )

    vertical_strength = _smooth_projection(
        vertical_strength,
        max(
            5,
            int(w * 0.008)
        )
    )

    strength_threshold = 0.18

    strong_columns = (
        vertical_strength
        >= strength_threshold
    )

    runs = _find_runs(
        strong_columns
    )

    minimum_length = (
        h
        * min_boundary_length_ratio
    )

    # Ignore the outer edge of the region.
    edge_margin = max(
        10,
        int(w * 0.03)
    )

    for start, end in runs:

        if (
            end - start
            < min_boundary_width
        ):
            continue

        if start < edge_margin:
            continue

        if end > w - edge_margin:
            continue

        centre = (
            start + end
        ) // 2

        # ----------------------------------------------------
        # Check whether the edge continues vertically.
        # ----------------------------------------------------

        local_gradient = gradient_x[
            :,
            max(0, start - 3):
            min(w, end + 3)
        ]

        row_strength = np.mean(
            local_gradient,
            axis=1
        )

        strong_rows = (
            row_strength
            >= gradient_threshold * 0.80
        )

        row_runs = _find_runs(
            strong_rows
        )

        longest = _longest_run_length(
            row_runs
        )

        if longest < minimum_length:
            continue

        confidence = min(
            1.0,
            longest / h
        )

        boundaries.append(
            {
                "orientation": "vertical",
                "position": int(
                    x + centre
                ),
                "start": int(y),
                "end": int(y + h),
                "width": int(
                    end - start
                ),
                "confidence": float(
                    confidence
                )
            }
        )

    # ========================================================
    # HORIZONTAL EDGES
    # ========================================================

    gradient_y = np.abs(
        cv2.Sobel(
            gray,
            cv2.CV_32F,
            0,
            1,
            ksize=3
        )
    )

    gradient_threshold_y = max(
        20.0,
        float(
            np.percentile(
                gradient_y,
                90
            )
        )
    )

    edge_pixels_y = (
        gradient_y
        >= gradient_threshold_y
    )

    horizontal_strength = np.mean(
        edge_pixels_y,
        axis=1
    )

    horizontal_strength = _smooth_projection(
        horizontal_strength,
        max(
            5,
            int(h * 0.008)
        )
    )

    strong_rows = (
        horizontal_strength
        >= strength_threshold
    )

    runs = _find_runs(
        strong_rows
    )

    minimum_length = (
        w
        * min_boundary_length_ratio
    )

    edge_margin = max(
        10,
        int(h * 0.03)
    )

    for start, end in runs:

        if (
            end - start
            < min_boundary_width
        ):
            continue

        if start < edge_margin:
            continue

        if end > h - edge_margin:
            continue

        centre = (
            start + end
        ) // 2

        # ----------------------------------------------------
        # Check whether the edge continues horizontally.
        # ----------------------------------------------------

        local_gradient = gradient_y[
            max(0, start - 3):
            min(h, end + 3),
            :
        ]

        column_strength = np.mean(
            local_gradient,
            axis=0
        )

        strong_columns = (
            column_strength
            >= gradient_threshold_y * 0.80
        )

        column_runs = _find_runs(
            strong_columns
        )

        longest = _longest_run_length(
            column_runs
        )

        if longest < minimum_length:
            continue

        confidence = min(
            1.0,
            longest / w
        )

        boundaries.append(
            {
                "orientation": "horizontal",
                "position": int(
                    y + centre
                ),
                "start": int(x),
                "end": int(x + w),
                "width": int(
                    end - start
                ),
                "confidence": float(
                    confidence
                )
            }
        )

    # --------------------------------------------------------
    # Strongest boundaries first.
    # --------------------------------------------------------

    boundaries.sort(
        key=lambda item: (
            item["confidence"],
            item["width"]
        ),
        reverse=True
    )

    diagnostics = {
        "vertical_candidates": sum(
            1
            for item in boundaries
            if item["orientation"]
            == "vertical"
        ),
        "horizontal_candidates": sum(
            1
            for item in boundaries
            if item["orientation"]
            == "horizontal"
        )
    }

    return (
        boundaries,
        diagnostics
    )


# ============================================================
# DRAW PAPER REGIONS
# ============================================================

def draw_paper_regions(
    image,
    regions
):
    """Draw likely paper regions in green."""

    preview = image.copy()

    for index, region in enumerate(
        regions,
        start=1
    ):

        x = region["x"]
        y = region["y"]
        w = region["width"]
        h = region["height"]

        cv2.rectangle(
            preview,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            5
        )

        cv2.putText(
            preview,
            f"P{index} "
            f"{region['score']:.2f}",
            (
                x + 10,
                max(
                    30,
                    y + 30
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return preview


# ============================================================
# DRAW INTERNAL BOUNDARIES
# ============================================================

def draw_internal_boundaries(
    image,
    boundaries
):
    """
    Draw possible physical paper boundaries.

    Blue  = vertical boundary
    Red   = horizontal boundary
    """

    preview = image.copy()

    for index, boundary in enumerate(
        boundaries,
        start=1
    ):

        confidence = (
            boundary["confidence"]
        )

        if (
            boundary["orientation"]
            == "vertical"
        ):

            x = boundary["position"]

            cv2.line(
                preview,
                (
                    x,
                    boundary["start"]
                ),
                (
                    x,
                    boundary["end"]
                ),
                (255, 0, 0),
                5
            )

            label_position = (
                x + 10,
                max(
                    30,
                    boundary["start"]
                    + 30
                )
            )

        else:

            y = boundary["position"]

            cv2.line(
                preview,
                (
                    boundary["start"],
                    y
                ),
                (
                    boundary["end"],
                    y
                ),
                (0, 0, 255),
                5
            )

            label_position = (
                max(
                    10,
                    boundary["start"]
                    + 10
                ),
                max(
                    30,
                    y - 10
                )
            )

        cv2.putText(
            preview,
            f"B{index} "
            f"{confidence:.2f}",
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

    return preview