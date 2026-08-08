import cv2


def contour_measurements(contour):
    """
    Calculate useful measurements for a contour.
    """

    area = cv2.contourArea(contour)

    x, y, width, height = cv2.boundingRect(contour)

    bounding_area = width * height

    if width == 0:
        aspect_ratio = 0
    else:
        aspect_ratio = height / width

    if bounding_area == 0:
        fill_ratio = 0
    else:
        fill_ratio = area / bounding_area

    # Perimeter of the contour
    perimeter = cv2.arcLength(
        contour,
        True
    )

    # Minimum rotated rectangle
    rotated_rect = cv2.minAreaRect(contour)

    rotated_width, rotated_height = rotated_rect[1]
    angle = rotated_rect[2]

    rotated_area = (
        rotated_width *
        rotated_height
    )

    if rotated_area == 0:
        rectangularity = 0
    else:
        rectangularity = area / rotated_area

    # Orientation-independent aspect ratio
    if (
        rotated_width == 0 or
        rotated_height == 0
    ):
        orientation_aspect_ratio = 0
    else:
        long_side = max(
            rotated_width,
            rotated_height
        )

        short_side = min(
            rotated_width,
            rotated_height
        )

        orientation_aspect_ratio = (
            long_side / short_side
        )

    return {
        "area": area,
        "width": width,
        "height": height,
        "bounding_area": bounding_area,
        "aspect_ratio": aspect_ratio,
        "orientation_aspect_ratio": orientation_aspect_ratio,
        "fill_ratio": fill_ratio,
        "perimeter": perimeter,
        "rectangularity": rectangularity,
        "angle": angle,
    }