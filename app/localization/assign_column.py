# localization/assign_column.py

def assign_columns(products, x_overlap_threshold=0.3, debug=False):
    """
    Assign column numbers to products within each shelf row.
    Products that overlap horizontally are grouped into the same column.

    Parameters
    ----------
    products : list[dict]
        List of product detections with 'bbox' and 'row'.
    x_overlap_threshold : float
        Minimum ratio of horizontal overlap to consider products in the same column (0–1).
        Example: 0.3 means at least 30% overlap in X axis.
    debug : bool
        If True, prints grouping information for debugging.

    Returns
    -------
    list[dict]
        Updated list of products with 'col' assigned.
    """
    if not products:
        return products

    # Group products by row
    rows = {}
    for p in products:
        row = p.get("row", 0)
        rows.setdefault(row, []).append(p)

    for row, items in rows.items():
        # Sort products from left to right by center X
        items_sorted = sorted(items, key=lambda p: (p["bbox"][0] + p["bbox"][2]) / 2)

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

                # If small overlap, it's a new column
                if x_overlap_ratio < x_overlap_threshold:
                    col_idx += 1

                if debug:
                    print(f" - Compared with previous: overlap_x={x_overlap_ratio:.2f}, col={col_idx}")

            prod["col"] = col_idx
            prev_box = prod["bbox"]

    return products
