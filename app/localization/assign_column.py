# app/localization/assign_column.py

from typing import List, Dict


def assign_columns(
    products: List[Dict],
    x_overlap_threshold: float = 0.3,
    debug: bool = False
) -> List[Dict]:
    """
    Assign column numbers to products within each shelf row.
    Products that overlap horizontally are grouped into the same column.

    Parameters
    ----------
    products : List[Dict]
        List of product detections containing:
        - 'bbox' (tuple[int, int, int, int])
        - 'row' (int)
    x_overlap_threshold : float
        Minimum horizontal overlap ratio (0–1) to consider
        two products in the same column.
    debug : bool
        If True, prints detailed grouping information.

    Returns
    -------
    List[Dict]
        Updated list of products with:
        - 'col' (int) assigned per row.
    """
    if not products:
        return products

    # Group products by row
    rows: Dict[int, List[Dict]] = {}
    for p in products:
        row = p.get("row", 0)
        rows.setdefault(row, []).append(p)

    for row, items in rows.items():

        # Sort products from left to right by center X
        items_sorted = sorted(
            items,
            key=lambda p: (p["bbox"][0] + p["bbox"][2]) / 2
        )

        col_idx = 1
        prev_box = None

        if debug:
            print(f"\n[assign_columns] Processing row {row} with {len(items_sorted)} products")

        for prod in items_sorted:
            x1, y1, x2, y2 = prod["bbox"]
            width = x2 - x1

            if prev_box is not None:
                x1_prev, y1_prev, x2_prev, y2_prev = prev_box

                # Compute horizontal overlap ratio
                overlap_x = max(0, min(x2, x2_prev) - max(x1, x1_prev))
                min_width = min(width, x2_prev - x1_prev)
                x_overlap_ratio = overlap_x / min_width if min_width > 0 else 0

                # If overlap is below threshold → new column
                if x_overlap_ratio < x_overlap_threshold:
                    col_idx += 1

                if debug:
                    print(
                        f" - Compared with previous: "
                        f"overlap_ratio={x_overlap_ratio:.2f}, col={col_idx}"
                    )

            prod["col"] = col_idx
            prev_box = prod["bbox"]

    return products
