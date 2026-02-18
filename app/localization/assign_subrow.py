# app/localization/assign_subrow.py

from typing import List, Dict


def assign_subrows(
    products: List[Dict],
    debug: bool = False
) -> List[Dict]:
    """
    Assign subrow index to vertically stacked products within
    the same (row, col) group. Subrows are numbered from
    bottom (1) to top (N).

    Parameters
    ----------
    products : List[Dict]
        List of product detections containing:
        - 'bbox' (tuple[int, int, int, int])
        - 'row' (int)
        - 'col' (int)
    debug : bool
        If True, prints detailed subrow assignment information.

    Returns
    -------
    List[Dict]
        Updated list of products with:
        - 'subrow' (int) assigned per (row, col) group.
    """
    if not products:
        return products

    # Group products by (row, col)
    groups: Dict[tuple[int, int], List[Dict]] = {}
    for p in products:
        key = (p.get("row", 0), p.get("col", 0))
        groups.setdefault(key, []).append(p)

    for key, group in groups.items():
        row, col = key

        if len(group) <= 1:
            group[0]["subrow"] = 1
            continue

        # Sort products by vertical center (bottom → top)
        group_sorted = sorted(
            group,
            key=lambda p: (p["bbox"][1] + p["bbox"][3]) / 2,
            reverse=True
        )

        for idx, prod in enumerate(group_sorted, start=1):
            prod["subrow"] = idx

            if debug:
                print(
                    f"[assign_subrows] Row {row}, Col {col}: "
                    f"bbox={prod['bbox']} → Subrow {idx}"
                )

    return products
