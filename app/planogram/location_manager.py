from collections import defaultdict
from typing import Dict, List, Any

# Rules

def _rule_validate_spatial_position(
    det: Dict,
    expected_rows: set,
    expected_cols: set,
    expected_subrows: set,
    extras_at_start: int
) -> tuple[bool, int]:
    """
    RULE 1 - Spatial Position Validation

    Validates:
    - Row
    - Column (adjusted by unexpected products at row start)
    - Subrow

    Returns
    -------
    (is_valid_position, adjusted_col)
    """

    adjusted_col = det["col"] - extras_at_start

    is_valid = (
        det["row"] in expected_rows and
        adjusted_col in expected_cols and
        det["subrow"] in expected_subrows
    )

    return is_valid, adjusted_col


# Main Validation Function

def validate_location(
    best_label_per_id: Dict[Any, str],
    products_detected: List[Dict],
    label_ids_dict: Dict[str, Dict[str, Any]],
    unexpected_count_by_row: Dict[Any, int],
    results: Dict[str, Any],
    debug: bool
) -> None:
    """
    Validate detected product locations against expected planogram positions.
    """

    matches_by_label: Dict[str, List[Any]] = defaultdict(list)

    # Group IDs by matched label
    for pid, label in best_label_per_id.items():
        matches_by_label[label].append(pid)

    for label, ids in matches_by_label.items():
        expected = label_ids_dict.get(label)
        if not expected:
            continue

        expected_rows = set(expected.get("row", []))
        expected_cols = set(expected.get("col", []))
        expected_subrows = set(expected.get("subrow", []))

        ok_ids = []
        wrong_ids = []

        for pid in ids:
            det = next(p for p in products_detected if p["id"] == pid)

            extras = unexpected_count_by_row.get(det["row"], 0)

            # ---------------- RULE CALL ----------------
            is_valid, adjusted_col = _rule_validate_spatial_position(
                det,
                expected_rows,
                expected_cols,
                expected_subrows,
                extras
            )
            # -------------------------------------------

            if is_valid:
                ok_ids.append(pid)
            else:
                wrong_ids.append(pid)

            if debug:
                results["debug"].setdefault(str(pid), []).append({
                    "label": label,
                    "detected": {
                        "row": det["row"],
                        "col": det["col"],
                        "adjusted_col": adjusted_col,
                        "subrow": det["subrow"]
                    },
                    "expected": {
                        "row": list(expected_rows),
                        "col": list(expected_cols),
                        "subrow": list(expected_subrows)
                    },
                    "extras_at_start": extras,
                    "status": "match" if is_valid else "different_location"
                })

        if ok_ids:
            results["match"].append({label: ok_ids})

        if wrong_ids:
            results["different_location"].append({label: wrong_ids})
