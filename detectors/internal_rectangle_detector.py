"""
Internal receipt rectangle detector.

This detector is deliberately independent of the main splitter.

Purpose:
    When several pieces of paper touch each other, the normal contour
    detector may see them as one large object. This module attempts to
    recover individual receipt/document rectangles from that combined
    paper region.

Important:
    This module detects CANDIDATE RECTANGLES.
    It does not perform the actual splitting.

Strategy:
    1. Work inside a supplied paper region.
    2. Detect long, straight vertical and horizontal edges.
    3. Merge nearby/overlapping edge segments.
    4. Build rectangles from compatible pairs of vertical/horizontal edges.
    5. Score complete rectangles rather than individual lines.
    6. Reject rectangles which are too small, too thin, or poorly supported.
    7. Optionally save diagnostic images.

No changes to the production splitter are made here.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable, Optional
import json
import math
import os

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EdgeCandidate:
    orientation: str       # "vertical" or "horizontal"
    position: float
    start: float
    end: float
    support: float
    strength: float
    width: float = 1.0

    @property
    def length(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class RectangleCandidate:
    x: int
    y: int
    w: int
    h: int

    area_ratio: float
    aspect_ratio: float

    left_support: float
    right_support: float
    top_support: float
    bottom_support: float

    edge_support: float
    score: float

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULTS = {
    # Canny
    "canny_low": 40,
    "canny_high": 120,

    # Morphological closing
    "close_kernel": 7,

    # Minimum edge length as fraction of paper dimension
    "min_vertical_edge_fraction": 0.18,
    "min_horizontal_edge_fraction": 0.18,

    # Merge nearby lines
    "merge_position_fraction": 0.018,
    "merge_gap_fraction": 0.08,

    # Rectangle size
    "min_area_fraction": 0.035,
    "max_area_fraction": 0.98,

    # Receipt/document aspect ratios.
    #
    # We allow fairly broad values because receipts may be rotated,
    # photographed at perspective, folded, etc.
    "min_aspect": 0.18,
    "max_aspect": 5.50,

    # Required edge support
    "min_edge_support": 0.22,

    # Maximum number of rectangles returned
    "max_rectangles": 20,

    # Non-max suppression overlap
    "max_overlap": 0.55,
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _cfg(config: Optional[dict]) -> dict:
    result = dict(DEFAULTS)
    if config:
        result.update(config)
    return result


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("image is None")

    if len(image.shape) == 2:
        return image

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _normalise_region(region: Any, image_shape) -> tuple[int, int, int, int]:
    """
    Accept several common region representations.

    Supported:
        (x, y, w, h)
        [x, y, w, h]
        {"x": ..., "y": ..., "w": ..., "h": ...}
        {"x": ..., "y": ..., "width": ..., "height": ...}
    """

    height, width = image_shape[:2]

    if isinstance(region, dict):
        x = region.get("x", 0)
        y = region.get("y", 0)

        w = region.get("w", region.get("width"))
        h = region.get("h", region.get("height"))

        if w is None or h is None:
            raise ValueError(f"Invalid paper region: {region}")

    elif hasattr(region, "x") and hasattr(region, "y"):
        x = region.x
        y = region.y

        if hasattr(region, "w"):
            w = region.w
        else:
            w = region.width

        if hasattr(region, "h"):
            h = region.h
        else:
            h = region.height

    else:
        if len(region) != 4:
            raise ValueError(f"Invalid paper region: {region}")

        x, y, w, h = region

    x = max(0, int(round(x)))
    y = max(0, int(round(y)))
    w = int(round(w))
    h = int(round(h))

    w = min(w, width - x)
    h = min(h, height - y)

    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid clipped paper region: {(x, y, w, h)}")

    return x, y, w, h


def _line_overlap(a_start, a_end, b_start, b_end) -> float:
    overlap = max(0.0, min(a_end, b_end) - max(a_start, b_start))

    a_len = max(1.0, a_end - a_start)
    b_len = max(1.0, b_end - b_start)

    return overlap / min(a_len, b_len)


def _iou(a, b) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b

    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    if intersection == 0:
        return 0.0

    union = aw * ah + bw * bh - intersection

    return intersection / max(1, union)


# ---------------------------------------------------------------------------
# Edge extraction
# ---------------------------------------------------------------------------

def _build_edge_maps(gray: np.ndarray, config: dict):
    """
    Create edge maps emphasizing long receipt/document edges.
    """

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(
        blurred,
        config["canny_low"],
        config["canny_high"]
    )

    kernel_size = config["close_kernel"]

    kernel_h = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_size * 3, 1)
    )

    kernel_v = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, kernel_size * 3)
    )

    horizontal = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel_h
    )

    vertical = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel_v
    )

    return edges, horizontal, vertical


# ---------------------------------------------------------------------------
# Projection based long-edge detection
# ---------------------------------------------------------------------------

def _find_vertical_edges(
    edges: np.ndarray,
    config: dict
) -> list[EdgeCandidate]:

    height, width = edges.shape

    # Count edge pixels in each column.
    projection = np.sum(edges > 0, axis=0).astype(np.float32)

    # Smooth the projection so tiny text strokes do not dominate.
    smooth_width = max(3, int(width * 0.003))
    if smooth_width % 2 == 0:
        smooth_width += 1

    kernel = np.ones(smooth_width, dtype=np.float32) / smooth_width
    projection_smooth = np.convolve(
        projection,
        kernel,
        mode="same"
    )

    # A genuine vertical receipt edge should normally extend over a
    # substantial fraction of the paper height.
    threshold = height * config["min_vertical_edge_fraction"]

    mask = projection_smooth >= threshold

    groups = _groups_from_boolean_mask(mask)

    candidates = []

    for start, end in groups:
        position = (start + end) / 2.0
        length = end - start

        local_strength = float(
            np.max(projection_smooth[start:end + 1])
            / max(1.0, height)
        )

        support = min(
            1.0,
            length / max(1.0, height * 0.65)
        )

        candidates.append(
            EdgeCandidate(
                orientation="vertical",
                position=position,
                start=0,
                end=height,
                support=support,
                strength=local_strength,
                width=max(1.0, length),
            )
        )

    return candidates


def _find_horizontal_edges(
    edges: np.ndarray,
    config: dict
) -> list[EdgeCandidate]:

    height, width = edges.shape

    projection = np.sum(edges > 0, axis=1).astype(np.float32)

    smooth_width = max(3, int(height * 0.003))
    if smooth_width % 2 == 0:
        smooth_width += 1

    kernel = np.ones(smooth_width, dtype=np.float32) / smooth_width
    projection_smooth = np.convolve(
        projection,
        kernel,
        mode="same"
    )

    threshold = width * config["min_horizontal_edge_fraction"]

    mask = projection_smooth >= threshold

    groups = _groups_from_boolean_mask(mask)

    candidates = []

    for start, end in groups:
        position = (start + end) / 2.0
        length = end - start

        local_strength = float(
            np.max(projection_smooth[start:end + 1])
            / max(1.0, width)
        )

        support = min(
            1.0,
            length / max(1.0, width * 0.65)
        )

        candidates.append(
            EdgeCandidate(
                orientation="horizontal",
                position=position,
                start=0,
                end=width,
                support=support,
                strength=local_strength,
                width=max(1.0, length),
            )
        )

    return candidates


def _groups_from_boolean_mask(mask: np.ndarray):
    groups = []

    inside = False
    start = 0

    for i, value in enumerate(mask):
        if value and not inside:
            inside = True
            start = i

        elif not value and inside:
            groups.append((start, i - 1))
            inside = False

    if inside:
        groups.append((start, len(mask) - 1))

    return groups


# ---------------------------------------------------------------------------
# Hough line extraction
# ---------------------------------------------------------------------------

def _hough_edges(
    edges: np.ndarray,
    config: dict
) -> tuple[list[EdgeCandidate], list[EdgeCandidate]]:

    height, width = edges.shape

    min_line_length = int(
        min(width, height) * 0.12
    )

    max_gap = int(
        min(width, height) * 0.025
    )

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(30, int(min(width, height) * 0.03)),
        minLineLength=max(30, min_line_length),
        maxLineGap=max(10, max_gap),
    )

    vertical = []
    horizontal = []

    if lines is None:
        return vertical, horizontal

    for line in lines[:, 0]:
        x1, y1, x2, y2 = map(int, line)

        dx = x2 - x1
        dy = y2 - y1

        length = math.hypot(dx, dy)

        if length < min_line_length:
            continue

        angle = abs(math.degrees(math.atan2(dy, dx)))

        # Near-horizontal
        if angle < 8 or angle > 172:

            start = min(x1, x2)
            end = max(x1, x2)
            position = (y1 + y2) / 2.0

            horizontal.append(
                EdgeCandidate(
                    orientation="horizontal",
                    position=position,
                    start=start,
                    end=end,
                    support=length / max(1.0, width),
                    strength=1.0,
                )
            )

        # Near-vertical
        elif 82 < angle < 98:

            start = min(y1, y2)
            end = max(y1, y2)
            position = (x1 + x2) / 2.0

            vertical.append(
                EdgeCandidate(
                    orientation="vertical",
                    position=position,
                    start=start,
                    end=end,
                    support=length / max(1.0, height),
                    strength=1.0,
                )
            )

    return vertical, horizontal


# ---------------------------------------------------------------------------
# Merge edge candidates
# ---------------------------------------------------------------------------

def _merge_edges(
    edges: list[EdgeCandidate],
    dimension: int,
    config: dict
) -> list[EdgeCandidate]:

    if not edges:
        return []

    position_tolerance = max(
        3,
        int(dimension * config["merge_position_fraction"])
    )

    gap_tolerance = max(
        5,
        int(dimension * config["merge_gap_fraction"])
    )

    edges = sorted(edges, key=lambda e: e.position)

    merged = []

    for edge in edges:

        if not merged:
            merged.append(edge)
            continue

        previous = merged[-1]

        same_position = (
            abs(edge.position - previous.position)
            <= position_tolerance
        )

        overlapping = (
            edge.start <= previous.end + gap_tolerance
            and previous.start <= edge.end + gap_tolerance
        )

        if same_position and overlapping:

            total_support = previous.support + edge.support

            if total_support > 0:
                position = (
                    previous.position * previous.support
                    + edge.position * edge.support
                ) / total_support
            else:
                position = (
                    previous.position + edge.position
                ) / 2

            merged[-1] = EdgeCandidate(
                orientation=previous.orientation,
                position=position,
                start=min(previous.start, edge.start),
                end=max(previous.end, edge.end),
                support=max(
                    previous.support,
                    edge.support
                ),
                strength=max(
                    previous.strength,
                    edge.strength
                ),
                width=max(
                    previous.width,
                    edge.width
                ),
            )

        else:
            merged.append(edge)

    return merged


# ---------------------------------------------------------------------------
# Rectangle generation
# ---------------------------------------------------------------------------

def _rectangle_edge_support(
    edge_map: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int
) -> float:

    height, width = edge_map.shape

    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))

    # Sample slightly inside/outside the proposed edge.
    # This makes the test less sensitive to a one-pixel offset.
    samples = []

    if x1 == x2:
        for dx in (-2, -1, 0, 1, 2):
            xx = max(0, min(width - 1, x1 + dx))
            values = edge_map[y1:y2 + 1, xx]
            samples.append(np.mean(values > 0))

    elif y1 == y2:
        for dy in (-2, -1, 0, 1, 2):
            yy = max(0, min(height - 1, y1 + dy))
            values = edge_map[yy, x1:x2 + 1]
            samples.append(np.mean(values > 0))

    if not samples:
        return 0.0

    return float(max(samples))


def _build_rectangles(
    edge_map: np.ndarray,
    vertical_edges: list[EdgeCandidate],
    horizontal_edges: list[EdgeCandidate],
    config: dict
) -> list[RectangleCandidate]:

    height, width = edge_map.shape

    image_area = width * height

    rectangles = []

    for left in vertical_edges:
        for right in vertical_edges:

            if right.position <= left.position:
                continue

            x1 = int(round(left.position))
            x2 = int(round(right.position))

            rect_width = x2 - x1

            if rect_width < width * 0.10:
                continue

            for top in horizontal_edges:
                for bottom in horizontal_edges:

                    if bottom.position <= top.position:
                        continue

                    y1 = int(round(top.position))
                    y2 = int(round(bottom.position))

                    rect_height = y2 - y1

                    if rect_height < height * 0.10:
                        continue

                    area = rect_width * rect_height
                    area_ratio = area / image_area

                    if area_ratio < config["min_area_fraction"]:
                        continue

                    if area_ratio > config["max_area_fraction"]:
                        continue

                    aspect = rect_width / max(
                        1.0,
                        rect_height
                    )

                    # Permit either orientation.
                    normalised_aspect = max(
                        aspect,
                        1.0 / max(aspect, 1e-6)
                    )

                    if not (
                        config["min_aspect"]
                        <= aspect
                        <= config["max_aspect"]
                    ):
                        continue

                    # Measure actual edge support.
                    left_support = _rectangle_edge_support(
                        edge_map,
                        x1,
                        y1,
                        x1,
                        y2
                    )

                    right_support = _rectangle_edge_support(
                        edge_map,
                        x2,
                        y1,
                        x2,
                        y2
                    )

                    top_support = _rectangle_edge_support(
                        edge_map,
                        x1,
                        y1,
                        x2,
                        y1
                    )

                    bottom_support = _rectangle_edge_support(
                        edge_map,
                        x1,
                        y2,
                        x2,
                        y2
                    )

                    edge_support = (
                        left_support
                        + right_support
                        + top_support
                        + bottom_support
                    ) / 4.0

                    # Combine geometric edge support with the
                    # candidate-line support.
                    line_support = (
                        left.support
                        + right.support
                        + top.support
                        + bottom.support
                    ) / 4.0

                    combined_support = (
                        edge_support * 0.70
                        + min(1.0, line_support) * 0.30
                    )

                    if combined_support < config["min_edge_support"]:
                        continue

                    # Prefer rectangles with strong four-sided support.
                    score = (
                        combined_support * 70.0
                        + min(1.0, area_ratio / 0.25) * 15.0
                        + min(1.0, left.support) * 3.75
                        + min(1.0, right.support) * 3.75
                        + min(1.0, top.support) * 3.75
                        + min(1.0, bottom.support) * 3.75
                    )

                    rectangles.append(
                        RectangleCandidate(
                            x=x1,
                            y=y1,
                            w=rect_width,
                            h=rect_height,
                            area_ratio=area_ratio,
                            aspect_ratio=aspect,
                            left_support=left_support,
                            right_support=right_support,
                            top_support=top_support,
                            bottom_support=bottom_support,
                            edge_support=combined_support,
                            score=min(100.0, score),
                        )
                    )

    return rectangles


# ---------------------------------------------------------------------------
# Non-max suppression
# ---------------------------------------------------------------------------

def _remove_duplicate_rectangles(
    rectangles: list[RectangleCandidate],
    config: dict
) -> list[RectangleCandidate]:

    rectangles = sorted(
        rectangles,
        key=lambda r: r.score,
        reverse=True
    )

    selected = []

    for candidate in rectangles:

        box = (
            candidate.x,
            candidate.y,
            candidate.w,
            candidate.h,
        )

        duplicate = False

        for existing in selected:

            existing_box = (
                existing.x,
                existing.y,
                existing.w,
                existing.h,
            )

            if _iou(box, existing_box) > config["max_overlap"]:
                duplicate = True
                break

        if not duplicate:
            selected.append(candidate)

        if len(selected) >= config["max_rectangles"]:
            break

    return selected


# ---------------------------------------------------------------------------
# Diagnostic drawing
# ---------------------------------------------------------------------------

def _draw_edges(
    image: np.ndarray,
    vertical_edges: list[EdgeCandidate],
    horizontal_edges: list[EdgeCandidate]
) -> np.ndarray:

    result = image.copy()

    for edge in vertical_edges:
        x = int(round(edge.position))

        cv2.line(
            result,
            (x, 0),
            (x, result.shape[0] - 1),
            (255, 0, 255),
            3
        )

    for edge in horizontal_edges:
        y = int(round(edge.position))

        cv2.line(
            result,
            (0, y),
            (result.shape[1] - 1, y),
            (255, 0, 255),
            3
        )

    return result


def _draw_rectangles(
    image: np.ndarray,
    rectangles: list[RectangleCandidate]
) -> np.ndarray:

    result = image.copy()

    for index, rect in enumerate(rectangles, start=1):

        cv2.rectangle(
            result,
            (rect.x, rect.y),
            (
                rect.x + rect.w,
                rect.y + rect.h,
            ),
            (0, 255, 0),
            5
        )

        label = (
            f"R{index} "
            f"{rect.score:.0f}"
        )

        cv2.putText(
            result,
            label,
            (rect.x + 5, max(30, rect.y + 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            3,
            cv2.LINE_AA
        )

    return result


def _save_debug(
    debug_dir: str,
    name: str,
    image: np.ndarray
):
    os.makedirs(debug_dir, exist_ok=True)

    path = os.path.join(
        debug_dir,
        name
    )

    cv2.imwrite(path, image)

    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_internal_rectangles(
    image: np.ndarray,
    paper_region: Any,
    config: Optional[dict] = None,
    debug_dir: Optional[str] = None,
) -> list[RectangleCandidate]:

    """
    Detect plausible individual rectangles inside one paper region.

    Parameters
    ----------
    image:
        Original BGR image.

    paper_region:
        Region containing the combined paper.

    config:
        Optional configuration overrides.

    debug_dir:
        Optional directory for diagnostic images and JSON.

    Returns
    -------
    list[RectangleCandidate]
    """

    config = _cfg(config)

    x, y, w, h = _normalise_region(
        paper_region,
        image.shape
    )

    crop = image[
        y:y + h,
        x:x + w
    ].copy()

    gray = _ensure_gray(crop)

    edges, horizontal_map, vertical_map = _build_edge_maps(
        gray,
        config
    )

    # Combine ordinary Canny edges with the directional maps.
    directional_edges = cv2.bitwise_or(
        horizontal_map,
        vertical_map
    )

    # Projection-derived candidates.
    projection_vertical = _find_vertical_edges(
        vertical_map,
        config
    )

    projection_horizontal = _find_horizontal_edges(
        horizontal_map,
        config
    )

    # Hough-derived candidates.
    hough_vertical, hough_horizontal = _hough_edges(
        edges,
        config
    )

    vertical_edges = _merge_edges(
        projection_vertical + hough_vertical,
        w,
        config
    )

    horizontal_edges = _merge_edges(
        projection_horizontal + hough_horizontal,
        h,
        config
    )

    rectangles = _build_rectangles(
        directional_edges,
        vertical_edges,
        horizontal_edges,
        config
    )

    rectangles = _remove_duplicate_rectangles(
        rectangles,
        config
    )

    # Convert local crop coordinates back to original-image coordinates.
    for rect in rectangles:
        rect.x += x
        rect.y += y

    # Diagnostics.
    if debug_dir:

        os.makedirs(
            debug_dir,
            exist_ok=True
        )

        _save_debug(
            debug_dir,
            "01_crop.png",
            crop
        )

        _save_debug(
            debug_dir,
            "02_edges.png",
            edges
        )

        _save_debug(
            debug_dir,
            "03_directional_edges.png",
            directional_edges
        )

        edge_image = _draw_edges(
            crop,
            vertical_edges,
            horizontal_edges
        )

        _save_debug(
            debug_dir,
            "04_candidate_edges.png",
            edge_image
        )

        # Draw rectangles in LOCAL coordinates.
        local_rectangles = []

        for rect in rectangles:
            local_rectangles.append(
                RectangleCandidate(
                    x=rect.x - x,
                    y=rect.y - y,
                    w=rect.w,
                    h=rect.h,
                    area_ratio=rect.area_ratio,
                    aspect_ratio=rect.aspect_ratio,
                    left_support=rect.left_support,
                    right_support=rect.right_support,
                    top_support=rect.top_support,
                    bottom_support=rect.bottom_support,
                    edge_support=rect.edge_support,
                    score=rect.score,
                )
            )

        rectangle_image = _draw_rectangles(
            crop,
            local_rectangles
        )

        _save_debug(
            debug_dir,
            "05_internal_rectangles.png",
            rectangle_image
        )

        # Also produce a full-image diagnostic.
        full_image = image.copy()

        for index, rect in enumerate(
            rectangles,
            start=1
        ):

            cv2.rectangle(
                full_image,
                (rect.x, rect.y),
                (
                    rect.x + rect.w,
                    rect.y + rect.h
                ),
                (0, 255, 0),
                5
            )

            cv2.putText(
                full_image,
                f"R{index} {rect.score:.0f}",
                (
                    rect.x + 5,
                    max(30, rect.y + 30)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                3,
                cv2.LINE_AA
            )

        _save_debug(
            debug_dir,
            "06_full_image_rectangles.png",
            full_image
        )

        json_path = os.path.join(
            debug_dir,
            "rectangles.json"
        )

        with open(
            json_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                [
                    r.to_dict()
                    for r in rectangles
                ],
                f,
                indent=2
            )

    return rectangles


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def detect_from_paper_regions(
    image: np.ndarray,
    paper_regions: Iterable[Any],
    debug_root: Optional[str] = None,
    config: Optional[dict] = None,
):
    """
    Run the detector over every paper region.

    Returns:
        list of dictionaries:

        {
            "paper_region": ...,
            "rectangles": [...]
        }
    """

    results = []

    for index, region in enumerate(
        paper_regions,
        start=1
    ):

        debug_dir = None

        if debug_root:
            debug_dir = os.path.join(
                debug_root,
                f"region_{index:02d}"
            )

        rectangles = detect_internal_rectangles(
            image=image,
            paper_region=region,
            config=config,
            debug_dir=debug_dir,
        )

        results.append(
            {
                "paper_region": region,
                "rectangles": rectangles,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Simple self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Test internal rectangle detection."
    )

    parser.add_argument(
        "image",
        help="Image to inspect"
    )

    parser.add_argument(
        "--x",
        type=int,
        default=0
    )

    parser.add_argument(
        "--y",
        type=int,
        default=0
    )

    parser.add_argument(
        "--w",
        type=int,
        default=None
    )

    parser.add_argument(
        "--h",
        type=int,
        default=None
    )

    parser.add_argument(
        "--debug",
        default="internal_rectangles"
    )

    args = parser.parse_args()

    image = cv2.imread(args.image)

    if image is None:
        raise SystemExit(
            f"Could not load image: {args.image}"
        )

    if args.w is None:
        args.w = image.shape[1] - args.x

    if args.h is None:
        args.h = image.shape[0] - args.y

    region = (
        args.x,
        args.y,
        args.w,
        args.h,
    )

    rectangles = detect_internal_rectangles(
        image,
        region,
        debug_dir=args.debug,
    )

    print()
    print("=" * 60)
    print("INTERNAL RECTANGLE DETECTOR")
    print("=" * 60)
    print(f"Image: {args.image}")
    print(
        f"Region: x={args.x}, y={args.y}, "
        f"w={args.w}, h={args.h}"
    )
    print()
    print(
        f"Rectangles found: {len(rectangles)}"
    )
    print()

    for i, rect in enumerate(
        rectangles,
        start=1
    ):

        print(
            f"Rectangle {i}: "
            f"x={rect.x}, "
            f"y={rect.y}, "
            f"w={rect.w}, "
            f"h={rect.h}, "
            f"aspect={rect.aspect_ratio:.3f}, "
            f"support={rect.edge_support:.3f}, "
            f"score={rect.score:.1f}"
        )

    print()
    print(
        f"Diagnostics: {os.path.abspath(args.debug)}"
    )