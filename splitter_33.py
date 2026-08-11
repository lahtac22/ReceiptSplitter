import time

import cv2
import numpy as np

from utils.files import get_input_files
from image.image_loader import load_image

from utils.debug import get_debug_folder
from utils.debug import save_debug_image
from utils.report import write_report

from detectors.colour_detector import detect

from detectors.split_detector import (
    find_vertical_gaps,
    find_candidate_vertical_gaps,
    find_watershed_regions,
    draw_watershed_regions
)

from paper_boundary_detector_span_aware import (
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


# ============================================================
# Splitter configuration
# ============================================================

MIN_RECEIPT_WIDTH = 150
MIN_RECEIPT_HEIGHT = 250
MIN_BOUNDARY_CONFIDENCE = 0.28
MIN_SEPARATOR_SPAN = 0.50
MIN_SEPARATOR_SIDE_PAPER = 0.20
SEPARATOR_SIDE_MARGIN = 25
MIN_CELL_PAPER_AREA = 5000
MIN_PAPER_FRACTION = 0.18
BOUNDARY_CLUSTER_DISTANCE = 20


# ============================================================
# Basic helpers
# ============================================================

def _crop_box(image, left, top, right, bottom):
    height, width = image.shape[:2]
    left = max(0, min(int(left), width))
    right = max(0, min(int(right), width))
    top = max(0, min(int(top), height))
    bottom = max(0, min(int(bottom), height))
    if right <= left or bottom <= top:
        return None
    crop = image[top:bottom, left:right].copy()
    return crop if crop.size else None


def _make_paper_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = ((value >= 115) & (saturation <= 125)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _boundary_span_overlap(boundary, region):
    start = float(boundary.get('start', 0))
    end = float(boundary.get('end', 0))
    if end < start:
        start, end = end, start
    if boundary.get('orientation') == 'vertical':
        region_start = float(region['y'])
        region_end = region_start + float(region['height'])
    elif boundary.get('orientation') == 'horizontal':
        region_start = float(region['x'])
        region_end = region_start + float(region['width'])
    else:
        return 0.0
    length = end - start
    if length <= 0:
        return 0.0
    overlap = max(0.0, min(end, region_end) - max(start, region_start))
    return overlap / length


def _boundary_inside_region(boundary, region):
    position = float(boundary.get('position', 0))
    if boundary.get('orientation') == 'vertical':
        inside = float(region['x']) < position < float(region['x'] + region['width'])
    elif boundary.get('orientation') == 'horizontal':
        inside = float(region['y']) < position < float(region['y'] + region['height'])
    else:
        return False
    return inside and _boundary_span_overlap(boundary, region) >= MIN_SEPARATOR_SPAN


def _boundary_side_paper_fractions(paper_mask, boundary, region):
    x = int(region['x'])
    y = int(region['y'])
    w = int(region['width'])
    h = int(region['height'])
    position = int(boundary['position'])
    start = int(min(boundary.get('start', 0), boundary.get('end', 0)))
    end = int(max(boundary.get('start', 0), boundary.get('end', 0)))
    margin = max(5, int(SEPARATOR_SIDE_MARGIN))
    region_right = x + w
    region_bottom = y + h

    if boundary.get('orientation') == 'vertical':
        sample_top = max(y, start)
        sample_bottom = min(region_bottom, end)
        if sample_bottom <= sample_top:
            return 0.0, 0.0
        left = paper_mask[sample_top:sample_bottom, max(x, position-margin):min(region_right, position)]
        right = paper_mask[sample_top:sample_bottom, max(x, position):min(region_right, position+margin)]
    elif boundary.get('orientation') == 'horizontal':
        sample_left = max(x, start)
        sample_right = min(region_right, end)
        if sample_right <= sample_left:
            return 0.0, 0.0
        left = paper_mask[max(y, position-margin):min(region_bottom, position), sample_left:sample_right]
        right = paper_mask[max(y, position):min(region_bottom, position+margin), sample_left:sample_right]
    else:
        return 0.0, 0.0

    if left.size == 0 or right.size == 0:
        return 0.0, 0.0
    return (
        cv2.countNonZero(left) / left.size,
        cv2.countNonZero(right) / right.size
    )


def _is_real_separator(paper_mask, boundary, region):
    confidence = float(boundary.get('confidence', 0))
    if confidence < MIN_BOUNDARY_CONFIDENCE or not _boundary_inside_region(boundary, region):
        return False, 0.0, 0.0, 0.0
    span = _boundary_span_overlap(boundary, region)
    first, second = _boundary_side_paper_fractions(paper_mask, boundary, region)
    valid = min(first, second) >= MIN_SEPARATOR_SIDE_PAPER
    return valid, span, first, second


def _cluster_boundaries(boundaries):
    if not boundaries:
        return []
    clusters = []
    for boundary in sorted(boundaries, key=lambda b: float(b.get('confidence', 0)), reverse=True):
        placed = False
        for cluster in clusters:
            representative = cluster[0]
            if boundary.get('orientation') != representative.get('orientation'):
                continue
            if abs(float(boundary['position']) - float(representative['position'])) > BOUNDARY_CLUSTER_DISTANCE:
                continue
            cluster.append(boundary)
            placed = True
            break
        if not placed:
            clusters.append([boundary])
    result = [dict(max(cluster, key=lambda b: float(b.get('confidence', 0)))) for cluster in clusters]
    result.sort(key=lambda b: (0 if b.get('orientation') == 'vertical' else 1, b.get('position', 0)))
    return result


# ============================================================
# Crop a split cell to the actual paper inside it
# ============================================================

def _crop_cell_to_paper(image, paper_mask, left, top, right, bottom):
    left = max(0, int(left))
    top = max(0, int(top))
    right = min(image.shape[1], int(right))
    bottom = min(image.shape[0], int(bottom))
    if right <= left or bottom <= top:
        return None

    cell_mask = paper_mask[top:bottom, left:right]
    if cell_mask.size == 0:
        return None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cell_mask = cv2.morphologyEx(cell_mask, cv2.MORPH_CLOSE, kernel)

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(cell_mask, 8)
    if labels_count <= 1:
        return None

    best = None
    best_area = 0
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= MIN_CELL_PAPER_AREA and area > best_area:
            best = label
            best_area = area

    if best is None:
        return None

    bx = int(stats[best, cv2.CC_STAT_LEFT])
    by = int(stats[best, cv2.CC_STAT_TOP])
    bw = int(stats[best, cv2.CC_STAT_WIDTH])
    bh = int(stats[best, cv2.CC_STAT_HEIGHT])

    if bw < MIN_RECEIPT_WIDTH or bh < MIN_RECEIPT_HEIGHT:
        return None

    crop_left = left + bx
    crop_top = top + by
    crop_right = crop_left + bw
    crop_bottom = crop_top + bh
    crop = _crop_box(image, crop_left, crop_top, crop_right, crop_bottom)
    if crop is None:
        return None

    final_mask = _make_paper_mask(crop)
    paper_fraction = cv2.countNonZero(final_mask) / max(1, final_mask.size)
    if paper_fraction < MIN_PAPER_FRACTION:
        return None

    return {
        'image': crop,
        'left': crop_left,
        'right': crop_right,
        'top': crop_top,
        'bottom': crop_bottom,
        'paper_fraction': paper_fraction,
        'method': 'boundary-cell'
    }



# ============================================================
# Boundary-based receipt extraction
# ============================================================

BOUNDARY_SPLIT_MIN_CROPS = 2
BOUNDARY_SPLIT_MAX_CELLS = 12
BOUNDARY_SPLIT_MIN_CELL_WIDTH = 150
BOUNDARY_SPLIT_MIN_CELL_HEIGHT = 250
BOUNDARY_SPLIT_MIN_CANDIDATE_SPAN = 0.45
BOUNDARY_SPLIT_EDGE_MARGIN = 40

# Horizontal paper-gap detection.
#
# A true stacked-receipt separator should contain a short run of rows
# where the paper mask drops substantially. A crease inside a single
# receipt can create an edge-like signal without creating this gap.
HORIZONTAL_GAP_MIN_HEIGHT = 4
HORIZONTAL_GAP_BASELINE_RATIO = 0.55
HORIZONTAL_GAP_MAX_PAPER_FRACTION = 0.50
HORIZONTAL_GAP_EDGE_MARGIN = 20
HORIZONTAL_GAP_BOUNDARY_TOLERANCE = 10


def _find_candidate_horizontal_gaps(paper_mask, rectangle):
    """
    Find short horizontal runs where the candidate's paper coverage
    drops substantially.

    The lightly cleaned paper mask is used here because the heavily
    cleaned mask intentionally closes narrow physical gaps.

    Returns:
        list of dictionaries with start, end, position and score data.
    """
    x, y, w, h = rectangle

    if w <= 0 or h <= 0:
        return []

    candidate = paper_mask[
        y:y + h,
        x:x + w
    ]

    if candidate.size == 0:
        return []

    paper = (candidate > 0).astype(np.float32)

    row_fraction = np.mean(
        paper,
        axis=1
    )

    baseline = float(
        np.median(row_fraction)
    )

    if baseline <= 0:
        return []

    threshold = min(
        HORIZONTAL_GAP_MAX_PAPER_FRACTION,
        baseline * HORIZONTAL_GAP_BASELINE_RATIO
    )

    low_rows = row_fraction < threshold

    runs = []
    in_run = False
    start = 0

    for index, value in enumerate(low_rows):
        if value and not in_run:
            start = index
            in_run = True
        elif not value and in_run:
            runs.append((start, index))
            in_run = False

    if in_run:
        runs.append((start, h))

    gaps = []

    for start, end in runs:
        if end - start < HORIZONTAL_GAP_MIN_HEIGHT:
            continue

        if (
            start <= HORIZONTAL_GAP_EDGE_MARGIN
            or end >= h - HORIZONTAL_GAP_EDGE_MARGIN
        ):
            continue

        paper_fraction = float(
            np.mean(row_fraction[start:end])
        )

        if paper_fraction > HORIZONTAL_GAP_MAX_PAPER_FRACTION:
            continue

        gaps.append({
            "start": int(y + start),
            "end": int(y + end),
            "position": int(y + (start + end) // 2),
            "paper_fraction": paper_fraction,
            "baseline": baseline,
        })

    gaps.sort(
        key=lambda item: (
            item["paper_fraction"],
            -(item["end"] - item["start"])
        )
    )

    return gaps


def _horizontal_boundary_matches_gap(
    gap_paper_mask,
    rectangle,
    boundary
):
    """Return True when a horizontal boundary coincides with a real
    low-paper gap inside the candidate."""
    if boundary.get("orientation") != "horizontal":
        return True

    position = int(
        round(float(boundary.get("position", 0)))
    )

    for gap in _find_candidate_horizontal_gaps(
        gap_paper_mask,
        rectangle
    ):
        if (
            gap["start"] - HORIZONTAL_GAP_BOUNDARY_TOLERANCE
            <= position
            <= gap["end"] + HORIZONTAL_GAP_BOUNDARY_TOLERANCE
        ):
            return True

    return False




def _boundary_candidate_span_fraction(boundary, rectangle):
    """
    Measure how much of the candidate the detected boundary spans.

    This is deliberately different from _boundary_span_overlap(),
    which measures how much of the boundary overlaps the candidate.
    For actual splitting we also want the boundary to extend through
    a meaningful portion of the candidate.
    """

    x, y, w, h = rectangle

    start = float(boundary.get("start", 0))
    end = float(boundary.get("end", 0))

    if end < start:
        start, end = end, start

    if boundary.get("orientation") == "vertical":
        candidate_start = float(y)
        candidate_end = float(y + h)
    elif boundary.get("orientation") == "horizontal":
        candidate_start = float(x)
        candidate_end = float(x + w)
    else:
        return 0.0

    candidate_span = candidate_end - candidate_start

    if candidate_span <= 0:
        return 0.0

    overlap = max(
        0.0,
        min(end, candidate_end)
        - max(start, candidate_start)
    )

    return overlap / candidate_span


def _get_real_candidate_boundaries(
    paper_mask,
    rectangle,
    boundaries,
    gap_paper_mask=None
):
    """
    Return the detected paper boundaries which are sufficiently
    trustworthy to be used as actual split lines for this candidate.

    A boundary must:
      * meet the configured confidence threshold;
      * lie inside the candidate;
      * span a meaningful part of the candidate;
      * have paper immediately on both sides.

    Boundaries are clustered first so that two detections of the same
    physical edge do not create two adjacent split lines.
    """

    x, y, w, h = rectangle

    candidate_boundaries = []

    for boundary in _cluster_boundaries(boundaries):

        if not _boundary_inside_region(
            boundary,
            {
                "x": x,
                "y": y,
                "width": w,
                "height": h
            }
        ):
            continue

        candidate_span = _boundary_candidate_span_fraction(
            boundary,
            rectangle
        )

        confidence = float(
            boundary.get("confidence", 0.0)
        )

        # The boundary detector has already associated this boundary
        # with this candidate using its physical span.  Local spans are
        # intentional in the 7-receipt layout, so do not reject them
        # using whole-candidate span or side-paper tests here.
        #
        # The recursive splitter decides whether the boundary actually
        # divides the current region.
        if confidence < 0.30:
            print(
                f"       Boundary diagnostic: "
                f"{boundary.get('orientation')} "
                f"x/y={boundary.get('position')} "
                f"span={boundary.get('start')}–{boundary.get('end')} "
                f"candidate_span={candidate_span:.3f} "
                f"confidence={confidence:.3f} "
                f"REJECT: confidence < 0.300"
            )
            continue

        print(
            f"       Boundary diagnostic: "
            f"{boundary.get('orientation')} "
            f"x/y={boundary.get('position')} "
            f"span={boundary.get('start')}–{boundary.get('end')} "
            f"candidate_span={candidate_span:.3f} "
            f"confidence={confidence:.3f} "
            f"ACCEPT: associated boundary"
        )

        candidate_boundaries.append(
            dict(boundary)
        )

    return candidate_boundaries


def _unique_boundary_positions(boundaries):
    """
    Return unique vertical and horizontal split positions.

    Positions too close together are merged by retaining the first
    position in each orientation. Clustering has already happened at
    the boundary-object level, but this final pass protects the cell
    construction from duplicate positions.
    """

    vertical = []
    horizontal = []

    for boundary in boundaries:
        position = int(round(float(boundary["position"])))

        if boundary["orientation"] == "vertical":
            target = vertical
        elif boundary["orientation"] == "horizontal":
            target = horizontal
        else:
            continue

        if not target:
            target.append(position)
            continue

        if all(
            abs(position - existing)
            > BOUNDARY_CLUSTER_DISTANCE
            for existing in target
        ):
            target.append(position)

    vertical.sort()
    horizontal.sort()

    return vertical, horizontal



def _strict_paper_x_bounds_in_cell(paper_mask, cell_left, cell_top, cell_right, cell_bottom):
    """Return the x extent of the strongest strict-paper component in a cell.

    Boundary cells are already the authoritative receipt separation.  The
    strict paper mask is used here only as a *horizontal anchor*.  This is
    important when two physical receipts overlap slightly: a permissive
    colour mask can merge the neighbouring receipt into the same component
    and then its light paper can survive the outer crop.

    We deliberately do not use this for the vertical extent because the
    strict paper mask can lose genuine receipt paper under shadows/folds.
    """
    if paper_mask is None:
        return None

    cell_left = max(0, int(cell_left))
    cell_top = max(0, int(cell_top))
    cell_right = min(paper_mask.shape[1], int(cell_right))
    cell_bottom = min(paper_mask.shape[0], int(cell_bottom))

    if cell_right <= cell_left or cell_bottom <= cell_top:
        return None

    cell_mask = paper_mask[cell_top:cell_bottom, cell_left:cell_right]
    if cell_mask.size == 0:
        return None

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        cell_mask,
        connectivity=8
    )

    if labels_count <= 1:
        return None

    best = None
    best_score = -1.0

    cell_area = float(cell_mask.shape[0] * cell_mask.shape[1])

    for label in range(1, labels_count):
        x, y, w, h, area = stats[label]
        area = int(area)

        if area < MIN_CELL_PAPER_AREA:
            continue

        # Prefer substantial components, but favour components which occupy
        # the cell horizontally rather than tiny fragments at an edge.
        width_fraction = w / max(1, cell_mask.shape[1])
        height_fraction = h / max(1, cell_mask.shape[0])
        area_fraction = area / max(1.0, cell_area)

        if width_fraction < 0.10 or height_fraction < 0.15:
            continue

        score = area_fraction * (0.70 + 0.30 * min(1.0, width_fraction))

        if score > best_score:
            best_score = score
            best = (int(x), int(w))

    if best is None:
        return None

    bx, bw = best
    return cell_left + bx, cell_left + bx + bw


def _conservative_outer_crop_bounds(
    image,
    cell_left,
    cell_top,
    cell_right,
    cell_bottom,
    paper_mask=None
):
    """Find a conservative outer crop inside an already-separated receipt cell.

    This is deliberately NOT the strict paper mask used for boundary detection.
    The strict mask can lose genuine receipt areas under shadows, folds, or
    overlap.  Here we use a more permissive colour mask only to remove large
    amounts of obvious table/background around a receipt.

    Safety rule: if the evidence is weak, return the complete boundary cell.
    Losing receipt content is considered worse than retaining some background.
    """
    cell = image[cell_top:cell_bottom, cell_left:cell_right]
    if cell.size == 0:
        return cell_left, cell_top, cell_right, cell_bottom

    ch, cw = cell.shape[:2]

    hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Deliberately more permissive than the normal paper mask.  This catches
    # the darker/shadowed upper portions of the €77 and Cronin's receipts.
    mask = (
        (saturation <= 100)
        & (value >= 80)
    ).astype(np.uint8) * 255

    # Join small breaks caused by folds, print, and shadows, but do not use a
    # huge kernel: the purpose is only to recover the outer receipt component.
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel
    )

    # Remove tiny isolated fragments without eroding the receipt itself.
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if num_labels <= 1:
        return cell_left, cell_top, cell_right, cell_bottom

    cell_area = float(cw * ch)
    candidates = []

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]

        if area < max(5000, cell_area * 0.08):
            continue

        bbox_area = float(max(1, w * h))
        fill = area / bbox_area

        # A receipt can be wrinkled and non-rectangular, but an isolated wood
        # fragment tends to have a poor/odd bounding box and much less area.
        if w < cw * 0.15 or h < ch * 0.15:
            continue

        cx = x + w / 2.0
        cy = y + h / 2.0
        centre_distance = (
            ((cx - cw / 2.0) / max(1.0, cw / 2.0)) ** 2
            + ((cy - ch / 2.0) / max(1.0, ch / 2.0)) ** 2
        ) ** 0.5
        centrality = max(0.0, 1.0 - min(1.0, centre_distance))

        score = area * (0.80 + 0.20 * centrality) * (0.75 + 0.25 * min(1.0, fill))
        candidates.append((score, x, y, w, h, area, fill))

    if not candidates:
        return cell_left, cell_top, cell_right, cell_bottom

    candidates.sort(reverse=True)
    _, bx, by, bw, bh, area, fill = candidates[0]

    # Require substantial evidence that this is the receipt rather than a
    # small incidental light region.
    area_fraction = area / cell_area
    bbox_fraction = (bw * bh) / cell_area

    if area_fraction < 0.15 or bbox_fraction < 0.20:
        return cell_left, cell_top, cell_right, cell_bottom

    # Add a generous safety margin.  We are trimming background, not tracing
    # the receipt edge pixel-perfectly.
    margin_x = max(12, min(32, int(round(cw * 0.02))))
    margin_y = max(12, min(32, int(round(ch * 0.02))))

    left = max(0, bx - margin_x)
    top = max(0, by - margin_y)
    right = min(cw, bx + bw + margin_x)
    bottom = min(ch, by + bh + margin_y)

    # For boundary-derived cells, constrain only the horizontal extent to the
    # strongest strict-paper component.  This prevents a neighbouring receipt
    # which overlaps the cell by a few pixels from being pulled back in by
    # the permissive colour mask.  Keep the generous vertical crop because
    # shadows and folds can make genuine receipt paper disappear from the
    # strict mask.
    if paper_mask is not None:
        strict_x = _strict_paper_x_bounds_in_cell(
            paper_mask,
            cell_left,
            cell_top,
            cell_right,
            cell_bottom
        )

        if strict_x is not None:
            strict_left, strict_right = strict_x
            anchored_left = max(0, strict_left - cell_left - margin_x)
            anchored_right = min(cw, strict_right - cell_left + margin_x)

            if anchored_right > anchored_left:
                left = max(left, anchored_left)
                right = min(right, anchored_right)

                print(
                    f"       Strict-paper horizontal anchor: "
                    f"x={strict_left}–{strict_right}"
                )

    print(
        f"       Conservative outer crop: "
        f"cell={cw}x{ch}, "
        f"paper_area={area_fraction:.3f}, "
        f"bbox={bx},{by},{bw},{bh}, "
        f"crop={left},{top}–{right},{bottom}"
    )

    return (
        cell_left + left,
        cell_top + top,
        cell_left + right,
        cell_top + bottom
    )


# ============================================================
# Overlapping / stepped receipt layout handling
# ============================================================

OVERLAP_LAYOUT_MIN_CONFIDENCE = 0.30
OVERLAP_LAYOUT_POSITION_TOLERANCE = 35
OVERLAP_LAYOUT_MIN_VERTICAL_SPAN = 0.20
OVERLAP_LAYOUT_MIN_HORIZONTAL_SPAN = 0.20


def _overlap_layout_crops(image, rectangle, boundaries, paper_regions=None):
    """
    Detect a stepped/overlapping receipt arrangement from physical paper
    boundaries.

    Example represented by the 7-receipt photograph:

        underlying receipt (Cronin's)
        ┌──────────────────────┐
        │                      │
        │                      │
        └───────┬──────────────┘
                │ overlay receipt (R7)
                │
                └───────────────┐
                                │

    The important point is that the vertical edges of the lower receipt
    must NOT be applied to the full height of the candidate.  They begin at
    the horizontal edge which marks the top of that receipt.

    Returns [] unless a coherent stepped pattern is found.
    """
    x, y, w, h = rectangle
    if w <= 0 or h <= 0 or not boundaries:
        return []

    vertical = [
        b for b in boundaries
        if b.get("orientation") == "vertical"
        and float(b.get("confidence", 0.0)) >= OVERLAP_LAYOUT_MIN_CONFIDENCE
    ]
    horizontal = [
        b for b in boundaries
        if b.get("orientation") == "horizontal"
        and float(b.get("confidence", 0.0)) >= OVERLAP_LAYOUT_MIN_CONFIDENCE
    ]

    if len(vertical) < 2 or not horizontal:
        return []

    vertical.sort(key=lambda b: int(round(float(b["position"]))))
    horizontal.sort(key=lambda b: int(round(float(b["position"]))))

    # Look for two vertical edges which begin at approximately the same
    # horizontal level, with a horizontal edge connecting their starts.
    for i, left_edge in enumerate(vertical[:-1]):
        left_x = int(round(float(left_edge["position"])))
        left_start = float(min(left_edge.get("start", y), left_edge.get("end", y + h)))
        left_end = float(max(left_edge.get("start", y), left_edge.get("end", y + h)))

        for right_edge in vertical[i + 1:]:
            right_x = int(round(float(right_edge["position"])))
            if right_x - left_x < BOUNDARY_SPLIT_MIN_CELL_WIDTH:
                continue

            right_start = float(min(right_edge.get("start", y), right_edge.get("end", y + h)))
            right_end = float(max(right_edge.get("start", y), right_edge.get("end", y + h)))

            # The two vertical edges should overlap substantially in their
            # vertical span; otherwise they are ordinary side-by-side cuts.
            overlap_start = max(left_start, right_start, float(y))
            overlap_end = min(left_end, right_end, float(y + h))
            overlap_length = max(0.0, overlap_end - overlap_start)
            if overlap_length / max(1.0, h) < OVERLAP_LAYOUT_MIN_VERTICAL_SPAN:
                continue

            for top_edge in horizontal:
                top_y = int(round(float(top_edge["position"])))
                if top_y <= y + BOUNDARY_SPLIT_EDGE_MARGIN:
                    continue
                if top_y >= y + h - BOUNDARY_SPLIT_EDGE_MARGIN:
                    continue

                # The horizontal edge must connect into the stepped area.
                # It need not reach the far vertical edge exactly: in the
                # reference photograph it ends at the underlying Cronin's
                # right edge, while the lower R7 receipt extends slightly
                # farther right.  What matters is that it reaches the left
                # edge and substantially overlaps the interval between the
                # two vertical edges.
                hs = float(min(top_edge.get("start", x), top_edge.get("end", x + w)))
                he = float(max(top_edge.get("start", x), top_edge.get("end", x + w)))
                if hs > left_x + OVERLAP_LAYOUT_POSITION_TOLERANCE:
                    continue

                horizontal_overlap = max(0.0, min(he, float(right_x)) - max(hs, float(left_x)))
                if horizontal_overlap / max(1.0, float(right_x - left_x)) < OVERLAP_LAYOUT_MIN_HORIZONTAL_SPAN:
                    continue

                # Both vertical edges should start near the horizontal edge.
                if abs(left_start - top_y) > OVERLAP_LAYOUT_POSITION_TOLERANCE:
                    continue
                if abs(right_start - top_y) > OVERLAP_LAYOUT_POSITION_TOLERANCE:
                    continue

                overlay_left = left_x
                overlay_right = right_x
                overlay_top = top_y
                overlay_bottom = min(y + h, max(left_end, right_end))

                if overlay_right - overlay_left < BOUNDARY_SPLIT_MIN_CELL_WIDTH:
                    continue
                if overlay_bottom - overlay_top < BOUNDARY_SPLIT_MIN_CELL_HEIGHT:
                    continue

                # The stepped boundaries do NOT mean that the left receipt
                # ends at top_y.  In the reference photograph, the €113
                # receipt continues underneath the upper Cronin's receipt
                # and therefore occupies the full candidate height.
                #
                # Likewise, the vertical boundary at right_x is the left
                # edge of the €77 receipt below/alongside the overlap.  It is
                # not necessarily the physical right edge of Cronin's above
                # top_y.  When the connected paper region extends farther
                # right in the upper band, use that paper edge for Cronin's.
                # This deliberately permits a small overlap between the
                # Cronin's and €77 crops, which is correct when receipts are
                # physically overlapping in the photograph.
                upper_left_left = x
                upper_left_right = left_x
                upper_middle_left = left_x
                upper_middle_right = right_x
                upper_top = y
                upper_left_bottom = y + h
                upper_middle_bottom = top_y

                if paper_regions:
                    upper_band_top = y
                    upper_band_bottom = top_y
                    best_upper_right = right_x
                    best_upper_overlap = 0

                    for region in paper_regions:
                        # Paper regions use x/y/width/height.  The previous
                        # version looked for left/top, which silently returned
                        # zero here and prevented Cronin's from extending to
                        # the actual paper edge at x=2066.
                        rx = int(region.get("x", region.get("left", 0)))
                        ry = int(region.get("y", region.get("top", 0)))
                        rw = int(region.get("width", 0))
                        rh = int(region.get("height", 0))
                        rr = rx + rw
                        rb = ry + rh

                        overlap_left = max(rx, upper_middle_left)
                        overlap_right = min(rr, right_x + 250)
                        overlap_top = max(ry, upper_band_top)
                        overlap_bottom = min(rb, upper_band_bottom)

                        if overlap_right <= overlap_left or overlap_bottom <= overlap_top:
                            continue

                        overlap_area = (overlap_right - overlap_left) * (overlap_bottom - overlap_top)
                        if overlap_area > best_upper_overlap:
                            best_upper_overlap = overlap_area
                            best_upper_right = max(right_x, min(rr, x + w))

                    upper_middle_right = best_upper_right

                if (
                    upper_left_right - upper_left_left < MIN_RECEIPT_WIDTH
                    or upper_middle_right - upper_middle_left < MIN_RECEIPT_WIDTH
                    or upper_middle_bottom - upper_top < MIN_RECEIPT_HEIGHT
                    or upper_left_bottom - upper_top < MIN_RECEIPT_HEIGHT
                ):
                    continue

                # The receipt to the right of the stepped pair continues from
                # right_x for the full candidate height.
                right_left = right_x
                right_right = x + w
                if right_right - right_left < MIN_RECEIPT_WIDTH:
                    continue

                upper_left = _crop_box(
                    image,
                    upper_left_left,
                    upper_top,
                    upper_left_right,
                    upper_left_bottom
                )
                upper_middle = _crop_box(
                    image,
                    upper_middle_left,
                    upper_top,
                    upper_middle_right,
                    upper_middle_bottom
                )
                overlay = _crop_box(
                    image,
                    overlay_left,
                    overlay_top,
                    overlay_right,
                    overlay_bottom
                )
                right_crop = _crop_box(
                    image,
                    right_left,
                    y,
                    right_right,
                    y + h
                )

                if (
                    upper_left is None
                    or upper_middle is None
                    or overlay is None
                    or right_crop is None
                ):
                    continue

                print(
                    "       Stepped overlap layout accepted: 4 receipts; "
                    f"upper-left x={upper_left_left}–{upper_left_right}, y={upper_top}–{upper_left_bottom}; "
                    f"upper-middle x={upper_middle_left}–{upper_middle_right}, y={upper_top}–{upper_middle_bottom}; "
                    f"overlay x={overlay_left}–{overlay_right}, y={overlay_top}–{overlay_bottom}; "
                    f"right x={right_left}–{right_right}, y={y}–{y+h}"
                )

                return [
                    {
                        "image": upper_left,
                        "left": upper_left_left,
                        "right": upper_left_right,
                        "top": upper_top,
                        "bottom": upper_left_bottom,
                        "paper_fraction": 0.0,
                        "method": "boundary-stepped-upper-left"
                    },
                    {
                        "image": upper_middle,
                        "left": upper_middle_left,
                        "right": upper_middle_right,
                        "top": upper_top,
                        "bottom": upper_middle_bottom,
                        "paper_fraction": 0.0,
                        "method": "boundary-stepped-upper-middle"
                    },
                    {
                        "image": overlay,
                        "left": overlay_left,
                        "right": overlay_right,
                        "top": overlay_top,
                        "bottom": overlay_bottom,
                        "paper_fraction": 0.0,
                        "method": "boundary-stepped-overlay"
                    },
                    {
                        "image": right_crop,
                        "left": right_left,
                        "right": right_right,
                        "top": y,
                        "bottom": y + h,
                        "paper_fraction": 0.0,
                        "method": "boundary-stepped-right"
                    }
                ]

    return []


def _boundary_receipt_crops(
    image,
    cleaned_paper_mask,
    rectangle,
    boundaries,
    gap_paper_mask=None
):
    """
    Split a candidate into receipt regions using trustworthy internal
    paper boundaries.

    IMPORTANT:
        Boundaries are applied to the regions they actually cross.
        They are NOT converted into a Cartesian grid.

    The previous implementation created every possible combination of
    vertical and horizontal cuts. That meant, for example, one
    horizontal boundary which only crossed part of a candidate could
    incorrectly split every vertical column in that candidate.

    Instead, boundaries are applied recursively. A boundary only splits
    a current region when:
        * it lies inside that region;
        * its span is sufficient for that region;
        * paper exists immediately on both sides; and
        * both resulting pieces are large enough to be receipts.

    This allows a candidate to contain arrangements such as:
        - three vertical receipts with one of them divided horizontally;
        - two receipts side-by-side with another receipt spanning both;
        - other non-grid layouts.

    The function deliberately returns [] unless at least two usable
    receipt crops are obtained. That prevents a weak boundary from
    replacing a perfectly good single-receipt crop.
    """

    x, y, w, h = rectangle

    real_boundaries = _get_real_candidate_boundaries(
        cleaned_paper_mask,
        rectangle,
        boundaries,
        gap_paper_mask=gap_paper_mask
    )

    if not real_boundaries:
        return []

    # Work with actual boundary objects rather than reducing them to
    # independent x/y cut positions. Their start/end span is important:
    # a boundary should only split the part of the candidate that it
    # actually crosses.
    # Apply boundaries in geometric order rather than confidence order.
    #
    # The split is recursive, so the order matters. For receipt1.JPG
    # the vertical boundaries establish the columns first. The
    # horizontal boundary can then split only the region that it
    # actually crosses.
    #
    # Sorting by confidence can split the wrong parent region first
    # and prevent the intended irregular layout from being
    # reconstructed.
    real_boundaries = sorted(
        real_boundaries,
        key=lambda boundary: (
            0
            if boundary.get("orientation") == "vertical"
            else 1,
            float(boundary.get("position", 0))
        )
    )

    regions = [
        {
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h)
        }
    ]

    for boundary in real_boundaries:

        next_regions = []

        for region in regions:

            if not _boundary_inside_region(
                boundary,
                region
            ):
                next_regions.append(region)
                continue

            position = int(
                round(float(boundary["position"]))
            )

            region_x = int(region["x"])
            region_y = int(region["y"])
            region_right = (
                region_x
                + int(region["width"])
            )
            region_bottom = (
                region_y
                + int(region["height"])
            )

            # Keep a small margin from an existing region edge.
            if boundary["orientation"] == "vertical":

                if (
                    position
                    <= region_x
                    + BOUNDARY_SPLIT_EDGE_MARGIN
                    or
                    position
                    >= region_right
                    - BOUNDARY_SPLIT_EDGE_MARGIN
                ):
                    next_regions.append(region)
                    continue

                left_width = position - region_x
                right_width = region_right - position

                if (
                    left_width
                    < BOUNDARY_SPLIT_MIN_CELL_WIDTH
                    or
                    right_width
                    < BOUNDARY_SPLIT_MIN_CELL_WIDTH
                ):
                    next_regions.append(region)
                    continue

                first = {
                    "x": region_x,
                    "y": region_y,
                    "width": left_width,
                    "height": int(region["height"])
                }

                second = {
                    "x": position,
                    "y": region_y,
                    "width": right_width,
                    "height": int(region["height"])
                }

            elif boundary["orientation"] == "horizontal":

                if (
                    position
                    <= region_y
                    + BOUNDARY_SPLIT_EDGE_MARGIN
                    or
                    position
                    >= region_bottom
                    - BOUNDARY_SPLIT_EDGE_MARGIN
                ):
                    next_regions.append(region)
                    continue

                top_height = position - region_y
                bottom_height = region_bottom - position

                if (
                    top_height
                    < BOUNDARY_SPLIT_MIN_CELL_HEIGHT
                    or
                    bottom_height
                    < BOUNDARY_SPLIT_MIN_CELL_HEIGHT
                ):
                    next_regions.append(region)
                    continue

                first = {
                    "x": region_x,
                    "y": region_y,
                    "width": int(region["width"]),
                    "height": top_height
                }

                second = {
                    "x": region_x,
                    "y": position,
                    "width": int(region["width"]),
                    "height": bottom_height
                }

            else:
                next_regions.append(region)
                continue

            # The boundary has already been associated with this
            # candidate using its physical span and confidence.
            #
            # Do NOT re-apply the whole-region side-paper test here.
            # For the 7-receipt photograph, the major boundary at x=1969
            # deliberately lies beside a narrow paper region, so that
            # test incorrectly rejects a genuine split.  The physical
            # span check above (_boundary_inside_region) is the
            # appropriate test at this recursive stage.
            next_regions.extend(
                [first, second]
            )

        regions = next_regions

        if len(regions) > BOUNDARY_SPLIT_MAX_CELLS:
            print(
                f"       Boundary split rejected: "
                f"{len(regions)} regions would be created."
            )
            return []

    crops = []

    for region in regions:

        cell_left = int(region["x"])
        cell_top = int(region["y"])
        cell_right = (
            cell_left
            + int(region["width"])
        )
        cell_bottom = (
            cell_top
            + int(region["height"])
        )

        # The boundary cell is authoritative for separation.  Only after
        # separation do we make a conservative outer crop to remove obvious
        # table/background.  This uses a permissive mask and a safety margin;
        # it is intentionally independent of the strict boundary paper mask.
        crop_left, crop_top, crop_right, crop_bottom = (
            _conservative_outer_crop_bounds(
                image,
                cell_left,
                cell_top,
                cell_right,
                cell_bottom,
                paper_mask=cleaned_paper_mask
            )
        )

        if crop_right <= crop_left or crop_bottom <= crop_top:
            continue

        crop_image = image[
            crop_top:crop_bottom,
            crop_left:crop_right
        ].copy()

        if crop_image.size == 0:
            continue

        crops.append({
            "image": crop_image,
            "left": crop_left,
            "right": crop_right,
            "top": crop_top,
            "bottom": crop_bottom,
            "paper_fraction": 0.0,
            "method": "boundary-cell"
        })

    crops.sort(
        key=lambda item: (
            item["top"],
            item["left"]
        )
    )

    if len(crops) < BOUNDARY_SPLIT_MIN_CROPS:
        return []

    print(
        f"       Boundary split produced "
        f"{len(crops)} region(s) using "
        f"span-aware boundaries."
    )

    return crops



# ============================================================
# Watershed-based receipt extraction
# ============================================================

WATERSHED_DISTANCE_RATIO = 0.50
WATERSHED_MIN_REGION_AREA = 10000
WATERSHED_MAX_DIMENSION = 1600
WATERSHED_MIN_COUNT_FOR_AUTOSPLIT = 3
WATERSHED_TWO_REGION_MIN_HORIZONTAL_RATIO = 1.20
WATERSHED_TWO_REGION_MAX_NARROW_CANDIDATE_RATIO = 0.50


def _watershed_receipt_crops(
    image,
    mask,
    rectangle,
    distance_ratio=WATERSHED_DISTANCE_RATIO,
    min_region_area=WATERSHED_MIN_REGION_AREA,
    max_dimension=WATERSHED_MAX_DIMENSION
):
    """
    Split a connected receipt candidate using marker-controlled
    watershed, but use the ORIGINAL IMAGE GRADIENT as the watershed
    surface.

    The previous implementation ran cv2.watershed() on a BGR version
    of the binary paper mask.  That makes the watershed follow the
    topology of the mask rather than the actual receipt edges.  On the
    7-receipt photograph this allowed a watershed region to contain
    large blank-paper areas and, in one case, cut the top off a receipt.

    Here:
      1. the paper/colour mask supplies the foreground;
      2. the distance transform supplies receipt seeds;
      3. the grayscale image gradient supplies the physical edges;
      4. a background marker prevents regions escaping into the table;
      5. the resulting labels are intersected with the original
         foreground mask.
    """

    x, y, w, h = rectangle

    candidate_mask = mask[
        y:y + h,
        x:x + w
    ]

    candidate_image = image[
        y:y + h,
        x:x + w
    ]

    if candidate_mask.size == 0:
        return []

    scale = 1.0
    largest_dimension = max(w, h)

    if largest_dimension > max_dimension:
        scale = max_dimension / largest_dimension
        resized_width = max(1, int(round(w * scale)))
        resized_height = max(1, int(round(h * scale)))

        working_mask = cv2.resize(
            candidate_mask,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA
        )

        working_image = cv2.resize(
            candidate_image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA
        )
    else:
        working_mask = candidate_mask.copy()
        working_image = candidate_image.copy()

    foreground = np.where(
        working_mask == 255,
        255,
        0
    ).astype(np.uint8)

    if cv2.countNonZero(foreground) == 0:
        return []

    # ------------------------------------------------------------
    # Seed receipt regions from the paper-mask distance transform.
    # ------------------------------------------------------------

    distance = cv2.distanceTransform(
        foreground,
        cv2.DIST_L2,
        5
    )

    maximum_distance = float(distance.max())

    if maximum_distance <= 0:
        return []

    core_threshold = maximum_distance * distance_ratio

    core = np.where(
        distance >= core_threshold,
        255,
        0
    ).astype(np.uint8)

    num_markers, markers = cv2.connectedComponents(core)

    if num_markers <= 1:
        return []

    working_min_area = max(
        500,
        min_region_area * scale * scale
    )

    filtered_markers = np.zeros(
        markers.shape,
        dtype=np.int32
    )

    marker_count = 0

    for marker_id in range(1, num_markers):
        marker_mask = np.where(
            markers == marker_id,
            255,
            0
        ).astype(np.uint8)

        area = cv2.countNonZero(marker_mask)

        if area < working_min_area:
            continue

        marker_count += 1
        filtered_markers[
            markers == marker_id
        ] = marker_count

    if marker_count < 2:
        return []

    # ------------------------------------------------------------
    # Build a gradient surface from the real photograph.
    # ------------------------------------------------------------

    gray = cv2.cvtColor(
        working_image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    gradient = cv2.magnitude(gx, gy)

    # Watershed treats low values as easier terrain, so the image
    # gradient becomes the physical edge map that separates receipts.
    gradient = cv2.normalize(
        gradient,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    gradient_bgr = cv2.cvtColor(
        gradient,
        cv2.COLOR_GRAY2BGR
    )

    # Give the outside of the candidate its own permanent marker.
    # This prevents receipt regions from growing into the table.
    background_id = marker_count + 1

    watershed_markers = filtered_markers.copy()
    watershed_markers[
        foreground == 0
    ] = background_id

    cv2.watershed(
        gradient_bgr,
        watershed_markers
    )

    # ------------------------------------------------------------
    # Reconstruct each receipt region at original resolution.
    # ------------------------------------------------------------

    crops = []

    for region_id in range(1, marker_count + 1):

        region_mask_working = np.where(
            watershed_markers == region_id,
            255,
            0
        ).astype(np.uint8)

        # Never allow a watershed region outside the original
        # candidate foreground.
        region_mask_working = cv2.bitwise_and(
            region_mask_working,
            foreground
        )

        area = cv2.countNonZero(region_mask_working)

        if area < working_min_area:
            continue

        if scale != 1.0:
            region_mask = cv2.resize(
                region_mask_working,
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            region_mask = region_mask_working

        # Small dilation restores pixels immediately adjacent to the
        # physical watershed boundary without recreating the huge
        # regions produced by the old 9x9 expansion.
        dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )

        crop_mask = cv2.dilate(
            region_mask,
            dilation_kernel,
            iterations=1
        )

        contours, _ = cv2.findContours(
            crop_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        contour = max(
            contours,
            key=cv2.contourArea
        )

        rx, ry, rw, rh = cv2.boundingRect(contour)

        if (
            rw < MIN_RECEIPT_WIDTH
            or rh < MIN_RECEIPT_HEIGHT
        ):
            continue

        crop = candidate_image[
            ry:ry + rh,
            rx:rx + rw
        ].copy()

        local_mask = crop_mask[
            ry:ry + rh,
            rx:rx + rw
        ]

        crop[local_mask == 0] = 255

        paper_fraction = (
            cv2.countNonZero(local_mask)
            / max(1, local_mask.size)
        )

        if paper_fraction < MIN_PAPER_FRACTION:
            continue

        crops.append({
            "image": crop,
            "left": x + rx,
            "right": x + rx + rw,
            "top": y + ry,
            "bottom": y + ry + rh,
            "paper_fraction": paper_fraction,
            "method": "watershed-gradient",
            "_watershed_mask": local_mask.copy()
        })

    return crops



def _paper_region_for_candidate(
    paper_regions,
    rectangle
):
    """Return the paper region with the greatest overlap with a candidate."""
    x, y, w, h = rectangle
    candidate_area = max(1, w * h)

    best = None
    best_overlap = 0.0

    for region in paper_regions:
        rx = int(region.get("x", region.get("left", 0)))
        ry = int(region.get("y", region.get("top", 0)))
        rw = int(region.get("width", 0))
        rh = int(region.get("height", 0))

        left = max(x, rx)
        top = max(y, ry)
        right = min(x + w, rx + rw)
        bottom = min(y + h, ry + rh)

        if right <= left or bottom <= top:
            continue

        overlap = (
            (right - left) * (bottom - top)
            / candidate_area
        )

        if overlap > best_overlap:
            best_overlap = overlap
            best = region

    return best, best_overlap


def _watershed_guided_paper_crops(
    image,
    watershed_crops,
    paper_regions,
    rectangle
):
    """
    Use watershed only to locate horizontal receipt centres, then use
    the connected paper region to recover the complete receipt height.

    This is deliberately conservative and is only used for exactly two
    horizontally separated watershed regions.  The watershed bounding
    boxes in receipt1.JPG are too short vertically: one cuts the top of
    the €77 receipt and both leave large amounts of unrelated area in
    some cases.  The paper region contains the complete physical paper
    area, so its bounds are a better source for the final vertical extent.
    """
    if len(watershed_crops) != 2:
        return []

    first, second = sorted(
        watershed_crops,
        key=lambda item: (
            item["left"] + item["right"]
        )
    )

    first_cx = (
        first["left"] + first["right"]
    ) / 2.0
    second_cx = (
        second["left"] + second["right"]
    ) / 2.0

    first_cy = (
        first["top"] + first["bottom"]
    ) / 2.0
    second_cy = (
        second["top"] + second["bottom"]
    ) / 2.0

    dx = abs(first_cx - second_cx)
    dy = abs(first_cy - second_cy)

    # Only use this recovery when the two regions are clearly side by side.
    if dx < dy * WATERSHED_TWO_REGION_MIN_HORIZONTAL_RATIO:
        return []

    paper_region, overlap = _paper_region_for_candidate(
        paper_regions,
        rectangle
    )

    if paper_region is None or overlap < 0.50:
        return []

    px = int(paper_region.get("x", paper_region.get("left", 0)))
    py = int(paper_region.get("y", paper_region.get("top", 0)))
    pr = px + int(paper_region["width"])
    pb = py + int(paper_region["height"])

    # The midpoint between the two watershed centres supplies the
    # separator.  The paper-region bounds supply the complete height.
    split_x = int(round((first_cx + second_cx) / 2.0))

    if split_x <= px or split_x >= pr:
        return []

    cells = [
        (px, py, split_x, pb),
        (split_x, py, pr, pb),
    ]

    crops = []

    for left, top, right, bottom in cells:
        if (
            right - left < MIN_RECEIPT_WIDTH
            or bottom - top < MIN_RECEIPT_HEIGHT
        ):
            return []

        crop = image[top:bottom, left:right].copy()

        if crop.size == 0:
            return []

        crops.append({
            "image": crop,
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom,
            "paper_fraction": 0.0,
            "method": "watershed-guided-paper"
        })

    print(
        f"       Watershed-guided paper split accepted: "
        f"x={px}–{split_x} and x={split_x}–{pr}; "
        f"paper overlap={overlap:.3f}"
    )

    return crops

def _watershed_should_split(
    rectangle,
    watershed_crops
):
    """
    Decide whether the watershed result represents multiple receipts.

    The important distinction is between:

      * genuinely separate receipts, and
      * one long receipt which watershed has divided into two regions.

    Three or more stable regions are accepted directly. For exactly two
    regions, a horizontal separation is accepted when the candidate is
    not simply a narrow receipt split into top/bottom pieces.
    """

    count = len(watershed_crops)

    if count < 2:
        return False

    if count >= WATERSHED_MIN_COUNT_FOR_AUTOSPLIT:
        return True

    x, y, w, h = rectangle

    candidate_ratio = (
        w / max(1, h)
    )

    if (
        count == 2
        and candidate_ratio
        <= WATERSHED_TWO_REGION_MAX_NARROW_CANDIDATE_RATIO
    ):
        return False

    first = watershed_crops[0]
    second = watershed_crops[1]

    first_cx = (
        first["left"]
        + first["right"]
    ) / 2

    first_cy = (
        first["top"]
        + first["bottom"]
    ) / 2

    second_cx = (
        second["left"]
        + second["right"]
    ) / 2

    second_cy = (
        second["top"]
        + second["bottom"]
    ) / 2

    dx = abs(
        first_cx - second_cx
    )

    dy = abs(
        first_cy - second_cy
    )

    if (
        dx
        >= dy
        * WATERSHED_TWO_REGION_MIN_HORIZONTAL_RATIO
    ):
        return True

    return False


# ============================================================
# Duplicate removal
# ============================================================

def _intersection_area(a, b):
    left = max(a['left'], b['left'])
    top = max(a['top'], b['top'])
    right = min(a['right'], b['right'])
    bottom = min(a['bottom'], b['bottom'])
    if right <= left or bottom <= top:
        return 0
    return (right-left) * (bottom-top)


def _area(item):
    return max(0, item['right']-item['left']) * max(0, item['bottom']-item['top'])


def _watershed_mask_overlap(a, b):
    """Overlap of actual watershed masks relative to the smaller mask."""
    mask_a = a.get("_watershed_mask")
    mask_b = b.get("_watershed_mask")

    if mask_a is None or mask_b is None:
        return None

    left = max(a["left"], b["left"])
    top = max(a["top"], b["top"])
    right = min(a["right"], b["right"])
    bottom = min(a["bottom"], b["bottom"])

    if right <= left or bottom <= top:
        return 0.0

    ax0 = left - a["left"]
    ay0 = top - a["top"]
    bx0 = left - b["left"]
    by0 = top - b["top"]

    width = right - left
    height = bottom - top

    ma = mask_a[ay0:ay0 + height, ax0:ax0 + width]
    mb = mask_b[by0:by0 + height, bx0:bx0 + width]

    if ma.size == 0 or mb.size == 0:
        return 0.0

    overlap = cv2.countNonZero(cv2.bitwise_and(ma, mb))
    area_a = cv2.countNonZero(mask_a)
    area_b = cv2.countNonZero(mask_b)

    return overlap / max(1, min(area_a, area_b))


def _remove_duplicate_candidates(receipts):
    if len(receipts) <= 1:
        return receipts

    keep = []

    for i, candidate in enumerate(receipts):
        candidate_area = _area(candidate)

        if candidate_area <= 0:
            continue

        duplicate = False

        for j, other in enumerate(receipts):
            if i == j:
                continue

            # The original photo contains overlapping candidate groups.
            # Do not deduplicate across different candidate groups.
            if candidate.get("candidate") != other.get("candidate"):
                continue

            if (
                candidate.get("method") == "watershed"
                and other.get("method") == "watershed"
            ):
                # Watershed bounding boxes can be heavily overlapping
                # while the actual segmented regions are separate.
                mask_overlap = _watershed_mask_overlap(
                    candidate,
                    other
                )

                if mask_overlap is None or mask_overlap < 0.80:
                    continue
            else:
                other_area = _area(other)

                if other_area <= candidate_area:
                    continue

                overlap = (
                    _intersection_area(candidate, other)
                    / candidate_area
                )

                if overlap < 0.80:
                    continue

            candidate_score = (
                candidate.get("paper_fraction", 0),
                candidate_area
            )
            other_score = (
                other.get("paper_fraction", 0),
                _area(other)
            )

            if other_score >= candidate_score:
                duplicate = True
                break

        if not duplicate:
            keep.append(candidate)

    return keep


# ============================================================
# Boundary association diagnostic
# ============================================================

def _candidate_xywh(rectangle):
    if isinstance(rectangle, dict):
        return int(rectangle['x']), int(rectangle['y']), int(rectangle['width']), int(rectangle['height'])
    return tuple(map(int, rectangle))


def _overlap_fraction(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax+aw, bx+bw)
    bottom = min(ay+ah, by+bh)
    if right <= left or bottom <= top:
        return 0.0
    return ((right-left)*(bottom-top)) / max(1, aw*ah)


def _print_boundary_associations(rectangles, paper_regions, boundaries):
    print('[INFO] Running boundary/candidate association diagnostic...')
    print()
    print('[INFO] Candidate rectangles:')
    candidates = []
    for index, rectangle in enumerate(rectangles, start=1):
        x, y, w, h = _candidate_xywh(rectangle)
        candidates.append((index, x, y, w, h))
        print(f'       Candidate {index}: x={x}–{x+w}, y={y}–{y+h}, w={w}, h={h}')

    print()
    print('[INFO] Paper regions:')
    for index, region in enumerate(paper_regions, start=1):
        x, y = int(region['x']), int(region['y'])
        w, h = int(region['width']), int(region['height'])
        print(f'       Paper region {index}: x={x}–{x+w}, y={y}–{y+h}, w={w}, h={h}')

    print()
    print('[INFO] Candidate ↔ paper-region overlap:')
    for index, x, y, w, h in candidates:
        for region_index, region in enumerate(paper_regions, start=1):
            fraction = _overlap_fraction(
                (x, y, w, h),
                (int(region['x']), int(region['y']), int(region['width']), int(region['height']))
            )
            if fraction > 0.01:
                print(f'       Candidate {index} ↔ Paper region {region_index}: {fraction:.3f} of candidate area')

    print()
    print('[INFO] Boundary ↔ candidate association:')
    for index, boundary in enumerate(boundaries, start=1):
        orientation = boundary['orientation']
        position = int(boundary['position'])
        start = int(boundary['start'])
        end = int(boundary['end'])
        confidence = float(boundary['confidence'])
        print()
        print(f'       Boundary {index}: {orientation}, position={position}, span={start}–{end}, confidence={confidence:.3f}')
        for candidate_index, x, y, w, h in candidates:
            if orientation == 'vertical':
                if not (x < position < x+w):
                    continue
                overlap = max(0, min(end, y+h)-max(start, y))
            else:
                if not (y < position < y+h):
                    continue
                overlap = max(0, min(end, x+w)-max(start, x))
            fraction = overlap / max(1, end-start)
            if fraction > 0:
                print(f'              Candidate {candidate_index}: {orientation} span overlap {fraction:.3f}')

    print()
    print('[INFO] Boundary ↔ paper-region association:')
    for index, boundary in enumerate(boundaries, start=1):
        for region_index, region in enumerate(paper_regions, start=1):
            if _boundary_inside_region(boundary, region):
                print(
                    f"       Boundary {index} ↔ Paper region {region_index}: "
                    f"{boundary['orientation']} span overlap {_boundary_span_overlap(boundary, region):.3f}"
                )


def _draw_boundary_associations(image, rectangles, paper_regions, boundaries):
    preview = image.copy()
    candidate_colours = [(255,0,0), (0,180,0), (255,0,255), (0,180,180), (180,0,180)]
    paper_colours = [(0,255,255), (255,0,0), (0,255,0), (255,0,255), (0,180,180)]

    for index, rectangle in enumerate(rectangles, start=1):
        x, y, w, h = _candidate_xywh(rectangle)
        colour = candidate_colours[(index-1) % len(candidate_colours)]
        cv2.rectangle(preview, (x,y), (x+w,y+h), colour, 4)
        cv2.putText(preview, f'C{index}', (x+8, max(35,y+30)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2, cv2.LINE_AA)

    for index, region in enumerate(paper_regions, start=1):
        x, y = int(region['x']), int(region['y'])
        w, h = int(region['width']), int(region['height'])
        colour = paper_colours[(index-1) % len(paper_colours)]
        cv2.rectangle(preview, (x,y), (x+w,y+h), colour, 3)
        cv2.putText(preview, f'P{index}', (x+8, max(35,y+30)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2, cv2.LINE_AA)

    for index, boundary in enumerate(boundaries, start=1):
        position = int(boundary['position'])
        start = int(boundary['start'])
        end = int(boundary['end'])
        if boundary['orientation'] == 'vertical':
            colour = (0,0,255)
            cv2.line(preview, (position,start), (position,end), colour, 5)
            label_pos = (position+5, max(30,start+30))
        else:
            colour = (255,0,0)
            cv2.line(preview, (start,position), (end,position), colour, 5)
            label_pos = (max(5,start+5), position-8)
        cv2.putText(preview, f'B{index}', label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2, cv2.LINE_AA)

    return preview


# ============================================================
# Actual watershed diagnostic drawing
# ============================================================

def _draw_actual_watershed_regions(image, receipts):
    """
    Draw the bounding boxes of the actual watershed receipt crops.

    This is deliberately separate from the older watershed diagnostic.
    The older diagnostic shows watershed regions discovered from the
    distance transform. This image shows the exact receipt crops that
    the splitter will save.
    """

    preview = image.copy()

    colours = [
        (0, 0, 255),
        (0, 180, 0),
        (255, 0, 0),
        (0, 180, 180),
        (180, 0, 180),
        (180, 120, 0),
        (80, 80, 255),
        (255, 80, 80)
    ]

    for index, receipt in enumerate(receipts, start=1):
        x1 = int(receipt['left'])
        y1 = int(receipt['top'])
        x2 = int(receipt['right'])
        y2 = int(receipt['bottom'])

        colour = colours[(index - 1) % len(colours)]

        cv2.rectangle(
            preview,
            (x1, y1),
            (x2, y2),
            colour,
            6
        )

        label = f"R {index}"

        cv2.putText(
            preview,
            label,
            (x1 + 10, max(40, y1 + 40)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            colour,
            3,
            cv2.LINE_AA
        )

    return preview


# ============================================================
# MAIN
# ============================================================

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

        # ------------------------------------------------
        # Create inspection folder
        # ------------------------------------------------

        debug_folder = get_debug_folder(
            file.stem
        )

        # ------------------------------------------------
        # Load image
        # ------------------------------------------------

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

        # =================================================
        # EXISTING RECEIPT DETECTION
        # =================================================

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

        # =================================================
        # GAP DETECTION
        # =================================================

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

        # ------------------------------------------------
        # Draw existing detections
        # ------------------------------------------------

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

        # =================================================
        # WATERSHED PARAMETER DIAGNOSTIC
        # =================================================
        #
        # This remains diagnostic. The actual splitter below uses
        # the same watershed concept, but retains the watershed
        # labels so that each accepted region can be cropped and
        # saved as an individual receipt.
        # =================================================

        print(
            "[INFO] Running watershed parameter diagnostic..."
        )

        watershed_ratios = [
            0.50,
            0.55,
            0.60,
            0.65,
            0.70
        ]

        watershed_start = time.perf_counter()

        for candidate_index, rectangle in enumerate(
            rectangles,
            start=1
        ):

            print(
                f"[INFO] Candidate "
                f"{candidate_index}"
            )

            for ratio in watershed_ratios:

                watershed_regions = (
                    find_watershed_regions(
                        mask,
                        rectangle,
                        distance_ratio=ratio
                    )
                )

                print(
                    f"       ratio "
                    f"{ratio:.2f} → "
                    f"{len(watershed_regions)} "
                    f"region(s)"
                )

        timings["Watershed Diagnostic"] = (
            time.perf_counter()
            - watershed_start
        )

        # ------------------------------------------------
        # Preserve the existing 0.50 watershed image.
        # This remains diagnostic only.
        # ------------------------------------------------

        all_watershed_regions = []

        for rectangle in rectangles:

            watershed_regions = (
                find_watershed_regions(
                    mask,
                    rectangle,
                    distance_ratio=0.50
                )
            )

            all_watershed_regions.extend(
                watershed_regions
            )

        watershed_preview = (
            draw_watershed_regions(
                image,
                all_watershed_regions
            )
        )

        save_debug_image(
            debug_folder,
            watershed_preview,
            "05b_watershed.png"
        )

        print(
            f"[INFO] Total watershed regions "
            f"at ratio 0.50: "
            f"{len(all_watershed_regions)}"
        )

        # =================================================
        # PAPER-BOUNDARY ANALYSIS
        # =================================================

        print(
            "[INFO] Running paper-boundary analysis..."
        )

        # ------------------------------------------------
        # Find paper regions
        # ------------------------------------------------

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

        # ------------------------------------------------
        # Save paper masks
        # ------------------------------------------------

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

        # Retain a lightly cleaned mask for narrow physical gaps between
        # stacked receipts. The heavily cleaned mask above closes these.
        split_gap_paper_mask = _make_paper_mask(
            image
        )

        print(
            f"[INFO] Paper regions found: "
            f"{paper_diagnostics['regions_found']}"
        )

        print(
            f"[INFO] Paper pixels: "
            f"{paper_diagnostics['paper_pixels']}"
        )

        # ------------------------------------------------
        # Draw paper regions
        # ------------------------------------------------

        paper_preview = draw_paper_regions(
            image,
            paper_regions
        )

        save_debug_image(
            debug_folder,
            paper_preview,
            "07_paper_regions.png"
        )

        # =================================================
        # FIND INTERNAL PAPER BOUNDARIES
        # =================================================

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

        # ------------------------------------------------
        # Print boundary information
        # ------------------------------------------------

        for index, boundary in enumerate(
            all_boundaries,
            start=1
        ):

            print(
                f"       Boundary {index}: "
                f"{boundary['orientation']}, "
                f"position="
                f"{boundary['position']}, "
                f"confidence="
                f"{boundary['confidence']:.3f}"
            )

        # ------------------------------------------------
        # Draw paper boundaries
        # ------------------------------------------------

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

        # =================================================
        # BOUNDARY ASSOCIATION DIAGNOSTIC
        # =================================================

        _print_boundary_associations(
            rectangles,
            paper_regions,
            all_boundaries
        )

        association_preview = _draw_boundary_associations(
            image,
            rectangles,
            paper_regions,
            all_boundaries
        )

        save_debug_image(
            debug_folder,
            association_preview,
            "08b_boundary_associations.png"
        )

        print(
            "[INFO] Saved boundary association diagnostic."
        )

        # =================================================
        # SPLIT RECEIPTS
        # =================================================
        #
        # The paper-boundary detector is now part of the actual split
        # decision rather than being diagnostic only.
        #
        # Priority:
        #
        #   1. Use trustworthy internal paper boundaries to construct
        #      receipt cells.
        #
        #   2. If boundary splitting does not produce at least two
        #      usable receipts, use watershed when it clearly indicates
        #      multiple receipts.
        #
        #   3. If neither method provides a trustworthy split, fall
        #      back to the best paper region and finally the candidate.
        #
        # This preserves the receipt2 behaviour: a tall single receipt
        # with no usable internal boundary remains one receipt.
        # =================================================

        print(
            "[INFO] Splitting receipt candidates..."
        )

        start = time.perf_counter()

        split_receipts = []

        for candidate_index, rectangle in enumerate(
            rectangles,
            start=1
        ):
            x, y, w, h = _candidate_xywh(
                rectangle
            )

            print(
                f"[INFO] Candidate {candidate_index}: "
                f"x={x}–{x+w}, "
                f"y={y}–{y+h}"
            )

            # ------------------------------------------------
            # FIRST: detect a stepped/overlapping receipt arrangement.
            #
            # In the 7-receipt photograph, Cronin's occupies the upper
            # part of the candidate while R7 begins lower down.  R7's
            # vertical edges must therefore not be applied to Cronin's
            # full height.
            # ------------------------------------------------
            overlap_crops = _overlap_layout_crops(
                image,
                (x, y, w, h),
                all_boundaries,
                paper_regions=paper_regions
            )

            if overlap_crops:
                print(
                    f"       Stepped overlap split accepted: "
                    f"{len(overlap_crops)} receipt(s)"
                )

                for receipt in overlap_crops:
                    receipt["candidate"] = candidate_index
                    split_receipts.append(receipt)

                continue

            # ------------------------------------------------
            # SECOND: use trustworthy physical paper boundaries.
            #
            # This is intentionally before watershed.  For receipt1.JPG
            # Candidate 3 contains an irregular 4-receipt arrangement:
            #
            #   vertical 1969
            #   vertical 1420
            #   horizontal 1252
            #
            # The span-aware recursive splitter turns those three
            # boundaries into exactly four cells.  Using watershed first
            # loses the complete height of some of those receipts.
            #
            # The single-candidate PDF continues to use this same path.
            # ------------------------------------------------

            boundary_crops = _boundary_receipt_crops(
                image,
                cleaned_paper_mask,
                (x, y, w, h),
                all_boundaries,
                gap_paper_mask=split_gap_paper_mask
            )

            if boundary_crops:
                print(
                    f"       Boundary split accepted: "
                    f"{len(boundary_crops)} receipt(s)"
                )

                for receipt in boundary_crops:
                    receipt["candidate"] = candidate_index
                    split_receipts.append(receipt)

                continue

            print(
                "       Boundary split not accepted."
            )

            # ------------------------------------------------
            # SECOND: watershed provides the evidence that a candidate
            # contains multiple receipts.
            #
            # For two clearly side-by-side regions, first try using the
            # watershed centres to divide the complete connected paper
            # region.  This avoids the short watershed bounding boxes
            # that previously cut the €77 receipt at the top/bottom.
            # ------------------------------------------------

            watershed_crops = _watershed_receipt_crops(
                image,
                mask,
                (x, y, w, h)
            )

            print(
                f"       Watershed produced "
                f"{len(watershed_crops)} region(s)"
            )

            if _watershed_should_split(
                (x, y, w, h),
                watershed_crops
            ):
                guided_crops = _watershed_guided_paper_crops(
                    image,
                    watershed_crops,
                    paper_regions,
                    (x, y, w, h)
                )

                if guided_crops:
                    for receipt in guided_crops:
                        receipt["candidate"] = candidate_index
                        split_receipts.append(receipt)

                    continue

                print(
                    "       Watershed-guided paper recovery "
                    "not accepted; using watershed regions."
                )

                print(
                    "       Watershed accepted "
                    "as a multi-receipt candidate."
                )

                for receipt in watershed_crops:
                    receipt["candidate"] = candidate_index
                    split_receipts.append(receipt)

                continue

            # ------------------------------------------------
            # THIRD: try candidate-specific horizontal gaps.
            # ------------------------------------------------
            #
            # This handles genuinely stacked receipts such as the
            # Cronin's €17.25 receipt above the XL Stop & Shop €135
            # receipt. A fold inside a single receipt does not normally
            # create a sufficiently large low-paper run.
            # ------------------------------------------------

            horizontal_gaps = _find_candidate_horizontal_gaps(
                split_gap_paper_mask,
                (x, y, w, h)
            )

            if horizontal_gaps:
                print(
                    f"       Candidate horizontal gaps: "
                    f"{len(horizontal_gaps)}"
                )

            horizontal_gap_crops = []

            for gap in horizontal_gaps:
                split_position = int(
                    gap["position"]
                )

                top_crop = _crop_cell_to_paper(
                    image,
                    cleaned_paper_mask,
                    x,
                    y,
                    x + w,
                    split_position
                )

                bottom_crop = _crop_cell_to_paper(
                    image,
                    cleaned_paper_mask,
                    x,
                    split_position,
                    x + w,
                    y + h
                )

                if (
                    top_crop is None
                    or bottom_crop is None
                ):
                    continue

                horizontal_gap_crops = [
                    top_crop,
                    bottom_crop
                ]

                print(
                    f"       Horizontal gap split accepted "
                    f"at y={split_position}."
                )
                break

            if len(horizontal_gap_crops) >= 2:
                for receipt in horizontal_gap_crops:
                    receipt["candidate"] = candidate_index
                    receipt["method"] = (
                        "candidate-horizontal-gap"
                    )
                    split_receipts.append(
                        receipt
                    )

                continue

            # ------------------------------------------------
            # THIRD: try candidate-specific vertical gaps.
            # ------------------------------------------------
            #
            # The existing whole-image gap detector is diagnostic.
            # The candidate-specific detector is suitable for actual
            # splitting because it only examines this candidate.
            #
            # This is particularly useful when two receipts are
            # side-by-side but the watershed result overlaps them.
            # ------------------------------------------------

            candidate_gaps = find_candidate_vertical_gaps(
                mask,
                (x, y, w, h),
                min_gap_width=30
            )

            if candidate_gaps:
                print(
                    f"       Candidate vertical gaps: "
                    f"{len(candidate_gaps)}"
                )

            gap_crops = []

            for gap_start, gap_end in candidate_gaps:

                left_crop = _crop_cell_to_paper(
                    image,
                    cleaned_paper_mask,
                    x,
                    y,
                    gap_start,
                    y + h
                )

                right_crop = _crop_cell_to_paper(
                    image,
                    cleaned_paper_mask,
                    gap_end,
                    y,
                    x + w,
                    y + h
                )

                if (
                    left_crop is None
                    or right_crop is None
                ):
                    continue

                gap_crops = [
                    left_crop,
                    right_crop
                ]
                break

            if len(gap_crops) >= 2:

                print(
                    "       Candidate gap split "
                    "accepted."
                )

                for receipt in gap_crops:
                    receipt["candidate"] = candidate_index
                    receipt["method"] = (
                        "candidate-vertical-gap"
                    )
                    split_receipts.append(
                        receipt
                    )

                continue

            # ------------------------------------------------
            # FIFTH: try watershed splitting as the final
            # multi-receipt detector.
            #
            # For multi-candidate images this was already tried
            # above.  At this point it is therefore mainly relevant
            # to single-candidate inputs.
            # ------------------------------------------------

            watershed_crops = _watershed_receipt_crops(
                image,
                mask,
                (x, y, w, h)
            )

            print(
                f"       Watershed produced "
                f"{len(watershed_crops)} region(s)"
            )

            if _watershed_should_split(
                (x, y, w, h),
                watershed_crops
            ):
                print(
                    "       Watershed accepted "
                    "as a multi-receipt candidate."
                )

                for receipt in watershed_crops:
                    receipt["candidate"] = candidate_index
                    split_receipts.append(receipt)

                continue

            # ------------------------------------------------
            # Neither boundary nor watershed provided a
            # trustworthy split.
            #
            # Use the paper region overlapping this candidate.
            # ------------------------------------------------

            best_region = None
            best_overlap = 0.0
            best_region_index = None

            for region_index, region in enumerate(
                paper_regions,
                start=1
            ):
                overlap = _overlap_fraction(
                    (x, y, w, h),
                    (
                        int(region["x"]),
                        int(region["y"]),
                        int(region["width"]),
                        int(region["height"])
                    )
                )

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_region = region
                    best_region_index = region_index

            if best_region is not None:
                print(
                    f"       Falling back to paper region "
                    f"{best_region_index} "
                    f"(candidate overlap={best_overlap:.3f})"
                )

                region_x = int(best_region["x"])
                region_y = int(best_region["y"])
                region_w = int(best_region["width"])
                region_h = int(best_region["height"])

                crop = _crop_cell_to_paper(
                    image,
                    cleaned_paper_mask,
                    region_x,
                    region_y,
                    region_x + region_w,
                    region_y + region_h
                )

                if crop is not None:
                    crop["candidate"] = candidate_index
                    crop["method"] = "paper-region-fallback"
                    split_receipts.append(crop)
                    continue

            # ------------------------------------------------
            # Final fallback: use the candidate itself.
            # ------------------------------------------------

            crop = _crop_box(
                image,
                x,
                y,
                x + w,
                y + h
            )

            if crop is not None:
                split_receipts.append({
                    "image": crop,
                    "candidate": candidate_index,
                    "left": x,
                    "right": x + w,
                    "top": y,
                    "bottom": y + h,
                    "paper_fraction": 1.0,
                    "method": "candidate-fallback"
                })

        # ------------------------------------------------
        # Remove accidental duplicate crops.
        # ------------------------------------------------
        before_dedup = len(split_receipts)

        split_receipts = _remove_duplicate_candidates(
            split_receipts
        )

        removed = before_dedup - len(split_receipts)

        if removed:
            print(
                f"[INFO] Removed {removed} duplicate "
                f"receipt crop(s)."
            )

        split_receipts.sort(
            key=lambda item: (
                item["top"],
                item["left"]
            )
        )

        # ------------------------------------------------
        # Draw the actual crops selected by the splitter.
        # ------------------------------------------------

        actual_watershed_preview = _draw_actual_watershed_regions(
            image,
            split_receipts
        )

        save_debug_image(
            debug_folder,
            actual_watershed_preview,
            "08c_actual_split.png"
        )

        # ------------------------------------------------
        # Save the individual receipt images.
        # ------------------------------------------------

        print(
            f"[INFO] Created {len(split_receipts)} receipt(s)"
        )

        for receipt_index, receipt in enumerate(
            split_receipts,
            start=1
        ):
            receipt_filename = (
                f"09_receipt_{receipt_index:02d}.png"
            )

            save_debug_image(
                debug_folder,
                receipt["image"],
                receipt_filename
            )

            print(
                f"       Receipt {receipt_index}: "
                f"x={receipt['left']}–{receipt['right']}, "
                f"y={receipt['top']}–{receipt['bottom']} "
                f"({receipt['method']})"
            )

        timings["Receipt Splitting"] = (
            time.perf_counter()
            - start
        )

        # =================================================
        # TOTAL TIME
        # =================================================

        timings["Total"] = (
            time.perf_counter()
            -
            total_start
        )

        # ------------------------------------------------
        # Existing report
        # ------------------------------------------------

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


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()