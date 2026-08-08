import cv2
import numpy as np


def find_vertical_gaps(
    mask,
    min_gap_width=20
):
    """
    Find vertical gaps in a binary mask.

    This is the existing whole-image gap detector.
    """

    white_pixels = np.sum(
        mask == 255,
        axis=0
    )

    threshold = mask.shape[0] * 0.05

    gaps = []

    in_gap = False
    gap_start = 0

    for x, count in enumerate(
        white_pixels
    ):

        if count < threshold:

            if not in_gap:
                gap_start = x
                in_gap = True

        else:

            if in_gap:

                if (
                    x - gap_start
                    >= min_gap_width
                ):

                    gaps.append(
                        (gap_start, x)
                    )

                in_gap = False

    if in_gap:

        if (
            mask.shape[1] - gap_start
            >= min_gap_width
        ):

            gaps.append(
                (
                    gap_start,
                    mask.shape[1]
                )
            )

    return gaps


def find_candidate_vertical_gaps(
    mask,
    rectangle,
    min_gap_width=20
):
    """
    Find vertical gaps inside a candidate
    rectangle.

    Unlike find_vertical_gaps(), this function
    only examines the pixels inside the supplied
    candidate region.

    Returns:
        A list of (start, end) gap positions
        using image coordinates.
    """

    x, y, w, h = rectangle

    candidate = mask[
        y:y + h,
        x:x + w
    ]

    if candidate.size == 0:
        return []

    white_pixels = np.sum(
        candidate == 255,
        axis=0
    )

    threshold = h * 0.05

    gaps = []

    in_gap = False
    gap_start = 0

    for local_x, count in enumerate(
        white_pixels
    ):

        if count < threshold:

            if not in_gap:
                gap_start = local_x
                in_gap = True

        else:

            if in_gap:

                if (
                    local_x - gap_start
                    >= min_gap_width
                ):

                    gaps.append(
                        (
                            x + gap_start,
                            x + local_x
                        )
                    )

                in_gap = False

    if in_gap:

        if (
            w - gap_start
            >= min_gap_width
        ):

            gaps.append(
                (
                    x + gap_start,
                    x + w
                )
            )

    return gaps


def find_watershed_regions(
    mask,
    rectangle,
    distance_ratio=0.50,
    min_region_area=10000,
    max_dimension=1600
):
    """
    Experimentally identify separate regions inside
    a candidate using a distance transform and
    watershed segmentation.

    This function is diagnostic only.

    Large candidates are resized before processing
    to keep the experiment reasonably fast.

    The returned rectangles are mapped back to the
    original image coordinates.

    Returns:
        A list of bounding rectangles:

            (x, y, width, height)

        using original image coordinates.
    """

    x, y, w, h = rectangle

    candidate = mask[
        y:y + h,
        x:x + w
    ]

    if candidate.size == 0:
        return []

    # ----------------------------
    # Resize large candidates
    # ----------------------------

    scale = 1.0

    largest_dimension = max(
        w,
        h
    )

    if largest_dimension > max_dimension:

        scale = (
            max_dimension
            / largest_dimension
        )

        resized_width = max(
            1,
            int(round(w * scale))
        )

        resized_height = max(
            1,
            int(round(h * scale))
        )

        candidate = cv2.resize(
            candidate,
            (
                resized_width,
                resized_height
            ),
            interpolation=cv2.INTER_AREA
        )

    # ----------------------------
    # Create binary foreground
    # ----------------------------

    foreground = np.where(
        candidate == 255,
        255,
        0
    ).astype(np.uint8)

    if cv2.countNonZero(
        foreground
    ) == 0:
        return []

    # ----------------------------
    # Distance transform
    # ----------------------------

    distance = cv2.distanceTransform(
        foreground,
        cv2.DIST_L2,
        5
    )

    maximum_distance = float(
        distance.max()
    )

    if maximum_distance <= 0:
        return []

    # ----------------------------
    # Find foreground cores
    # ----------------------------

    core_threshold = (
        maximum_distance
        * distance_ratio
    )

    core = np.where(
        distance >= core_threshold,
        255,
        0
    ).astype(np.uint8)

    # ----------------------------
    # Find core components
    # ----------------------------

    num_markers, markers = (
        cv2.connectedComponents(
            core
        )
    )

    if num_markers <= 1:
        return []

    marker_count = 0

    filtered_markers = np.zeros(
        markers.shape,
        dtype=np.int32
    )

    # Scale the minimum area threshold
    # to the resized image.

    working_min_area = (
        min_region_area
        * scale
        * scale
    )

    working_min_area = max(
        500,
        working_min_area
    )

    for marker_id in range(
        1,
        num_markers
    ):

        marker_mask = np.where(
            markers == marker_id,
            255,
            0
        ).astype(np.uint8)

        area = cv2.countNonZero(
            marker_mask
        )

        if area < working_min_area:
            continue

        marker_count += 1

        filtered_markers[
            markers == marker_id
        ] = marker_count

    if marker_count < 2:
        return []

    # ----------------------------
    # Prepare watershed image
    # ----------------------------

    candidate_bgr = cv2.cvtColor(
        candidate,
        cv2.COLOR_GRAY2BGR
    )

    filtered_markers[
        foreground == 0
    ] = 0

    # ----------------------------
    # Watershed
    # ----------------------------

    watershed_markers = (
        filtered_markers.copy()
    )

    cv2.watershed(
        candidate_bgr,
        watershed_markers
    )

    # ----------------------------
    # Extract resulting regions
    # ----------------------------

    regions = []

    for region_id in range(
        1,
        marker_count + 1
    ):

        region_mask = np.where(
            watershed_markers == region_id,
            255,
            0
        ).astype(np.uint8)

        area = cv2.countNonZero(
            region_mask
        )

        if area < working_min_area:
            continue

        contours, _ = cv2.findContours(
            region_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        contour = max(
            contours,
            key=cv2.contourArea
        )

        rx, ry, rw, rh = cv2.boundingRect(
            contour
        )

        # Convert working-image coordinates
        # back to original-image coordinates.

        original_x = int(
            round(rx / scale)
        )

        original_y = int(
            round(ry / scale)
        )

        original_w = int(
            round(rw / scale)
        )

        original_h = int(
            round(rh / scale)
        )

        regions.append(
            (
                x + original_x,
                y + original_y,
                original_w,
                original_h
            )
        )

    return regions


def draw_watershed_regions(
    image,
    regions
):
    """
    Draw diagnostic bounding boxes around
    watershed regions.

    This is visualisation only.

    The returned image is a copy of the
    supplied image.
    """

    preview = image.copy()

    # A fixed set of colours makes the diagnostic
    # images easy to compare between test runs.

    colours = [
        (0, 0, 255),
        (0, 180, 0),
        (255, 0, 0),
        (0, 180, 180),
        (180, 0, 180),
        (180, 120, 0)
    ]

    for index, region in enumerate(
        regions,
        start=1
    ):

        x, y, w, h = region

        colour = colours[
            (index - 1)
            % len(colours)
        ]

        cv2.rectangle(
            preview,
            (x, y),
            (x + w, y + h),
            colour,
            6
        )

        label = (
            f"WS {index}"
        )

        label_y = max(
            40,
            y + 45
        )

        cv2.putText(
            preview,
            label,
            (x + 10, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            colour,
            3,
            cv2.LINE_AA
        )

    return preview