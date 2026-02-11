# localization/assign_subrow.py

def assign_subrows(products, debug=False):
    """
    Assign subrow index for products stacked vertically within the same (row, col).
    Subrows are numbered from bottom (1) to top (N).

    Parameters
    ----------
    products : list[dict]
        List of product detections with 'bbox', 'row', and 'col'.
    debug : bool
        If True, prints subrow assignments for debugging.
    
    Returns
    -------
    list[dict]
        Updated list of products with 'subrow' assigned
    """
    if not products:
        return products

    # Group by (row, col)
    groups = {}
    for p in products:
        key = (p.get("row", 0), p.get("col", 0))
        groups.setdefault(key, []).append(p)

    for key, group in groups.items():
        row, col = key
        if len(group) <= 1:
            group[0]["subrow"] = 1
            continue

        # Sort by vertical center from bottom to top
        group_sorted = sorted(group, key=lambda p: (p["bbox"][1] + p["bbox"][3]) / 2, reverse=True)

        for idx, prod in enumerate(group_sorted, start=1):
            prod["subrow"] = idx
            if debug:
                print(f"[assign_subrows] Row {row}, Col {col}: bbox={prod['bbox']} → Subrow {idx}")

    return products
