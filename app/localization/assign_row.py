# localization/assign_row.py

def assign_rows(products, shelves, debug=False):
    """
    Assign each product to the shelf row directly below it.
    Rows are numbered from bottom (1) to top (N).

    Parameters
    ----------
    products : list[dict]
        List of product detections with 'bbox'.
    shelves : list[dict]
        List of shelf detections with 'bbox'.
    debug : bool
        If True, prints detailed information about the row assignment process.

    Returns
    -------
    list[dict]
        Updated list of products with 'row' assigned.
    """
    if not shelves or not products:
        return products

    # Sort shelves by bottom Y coordinate (descending = bottom first)
    shelves_sorted = sorted(shelves, key=lambda s: s["bbox"][3], reverse=True)

    if debug:
        print(f"\n[assign_rows] Found {len(shelves_sorted)} shelves (bottom→top)")

    for p in products:
        x1, y1, x2, y2 = p["bbox"]
        center_y = (y1 + y2) / 2
        row_assigned = None

        # Find the shelf directly below the product
        for i, shelf in enumerate(shelves_sorted):
            shelf_top = shelf["bbox"][1]  # top Y of the shelf
            shelf_bottom = shelf["bbox"][3]  # bottom Y of the shelf
            if shelf_bottom >= center_y >= shelf_top:
                # Product is within the shelf → assign corresponding row
                row_assigned = i + 1
                break
            elif center_y < shelf_top:
                # Product is above this shelf, keep looking higher
                continue
            elif center_y > shelf_bottom:
                # Product is above this shelf → this is the row below
                row_assigned = i + 1
                break

        # If the product is above the highest shelf
        if row_assigned is None:
            row_assigned = len(shelves_sorted)

        p["row"] = row_assigned

        if debug:
            print(f" - Product bbox={p['bbox']} center_y={center_y:.1f} → Row {row_assigned}")

    return products
