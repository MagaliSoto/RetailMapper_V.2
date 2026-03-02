import logging
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Set


logger = logging.getLogger(__name__)


# =====================================================
# RULES
# =====================================================

def _rule_validate_spatial_position(
    det: Dict,
    expected_rows: Set[int],
    expected_cols: Set[int],
    expected_subrows: Set[int],
    unexpected_cols_by_row: Dict[Any, List[int]],
    missing_positions_by_row: Dict[Any, List[int]],
) -> Tuple[bool, int]:
    """
    RULE 1 - Spatial Position Validation (Index-based adjustment)

    Validates whether a detected product is in a valid expected position.
    Instead of applying structural shift logic, this rule performs
    index-based alignment correction when missing or unexpected
    elements exist within the same row.

    Returns:
        Tuple:
            - bool: whether the position is valid
            - int: adjusted column index after alignment correction
    """

    row = det["row"]
    col = det["col"]

    logger.debug(
        "Validating spatial position for product ID %s at row=%s col=%s",
        det.get("id"),
        row,
        col
    )

    # Apply column alignment adjustment if structural inconsistencies exist
    if row in unexpected_cols_by_row or row in missing_positions_by_row:

        common_cols = list(
            set(unexpected_cols_by_row.get(row, [])) &
            set(missing_positions_by_row.get(row, []))
        )

        only_in_missing = list(
            set(missing_positions_by_row.get(row, [])) -
            set(unexpected_cols_by_row.get(row, []))
        )

        shift = sum(1 for mcol in only_in_missing if mcol <= col)
        adjusted_col = col + shift

        logger.debug(
            "Row %s adjustment applied. Original col=%s, adjusted col=%s",
            row,
            col,
            adjusted_col
        )

    else:
        adjusted_col = col

    is_valid = (
        row in expected_rows and
        adjusted_col in expected_cols and
        det["subrow"] in expected_subrows
    )

    logger.debug(
        "Validation result for product ID %s: %s",
        det.get("id"),
        "VALID" if is_valid else "INVALID"
    )

    return is_valid, adjusted_col


# =====================================================
# MAIN VALIDATION FUNCTION
# =====================================================

def validate_location(
    best_label_per_id: Dict[Any, str],
    products_detected: List[Dict],
    label_ids_dict: Dict[str, Dict[str, Any]],
    unexpected_cols_by_row: Dict[Any, List[int]],
    missing_positions_by_row: Dict[Any, List[int]],
    results: Dict[str, Any],
    debug: bool
) -> None:
    """
    Validates detected product locations against expected planogram positions.

    This function:
    - Groups detected products by label
    - Applies spatial validation rules
    - Separates matches and different_location results
    - Optionally stores detailed debug information
    """

    logger.info("Starting location validation")

    # Build expected column map per row
    expected_cols_by_row = defaultdict(set)

    for label_data in label_ids_dict.values():
        rows = label_data.get("row", [])
        cols = label_data.get("col", [])

        for row in rows:
            for col in cols:
                expected_cols_by_row[row].add(col)

    logger.debug("Expected columns by row constructed")

    # Group detected products by row (left to right order)
    products_by_row = defaultdict(list)

    for det in products_detected:
        products_by_row[det["row"]].append(det)

    for row in products_by_row:
        products_by_row[row] = sorted(
            products_by_row[row],
            key=lambda x: x["col"]
        )

    logger.debug("Products grouped and sorted by row")

    # Group product IDs by matched label
    matches_by_label: Dict[str, List[Any]] = defaultdict(list)

    for pid, label in best_label_per_id.items():
        matches_by_label[label].append(pid)

    logger.debug("Products grouped by matched label")

    # Validate each label group
    for label, ids in matches_by_label.items():

        expected = label_ids_dict.get(label)
        if not expected:
            logger.warning("Label '%s' not found in planogram definition", label)
            continue

        expected_rows = set(expected.get("row", []))
        expected_cols = set(expected.get("col", []))
        expected_subrows = set(expected.get("subrow", []))

        ok_ids = []
        wrong_ids = []

        for pid in ids:

            # Retrieve detection structure for this product ID
            det = next(p for p in products_detected if p["id"] == pid)

            # ---------------- RULE EXECUTION ----------------
            is_valid, adjusted_col = _rule_validate_spatial_position(
                det,
                expected_rows,
                expected_cols,
                expected_subrows,
                unexpected_cols_by_row,
                missing_positions_by_row
            )
            # ------------------------------------------------

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
                    "status": "match" if is_valid else "different_location"
                })

        # Append results
        if ok_ids:
            results["match"].append({label: ok_ids})
            logger.debug(
                "Label '%s': %d matches",
                label,
                len(ok_ids)
            )

        if wrong_ids:
            results["different_location"].append({label: wrong_ids})
            logger.debug(
                "Label '%s': %d different_location",
                label,
                len(wrong_ids)
            )

    logger.info("Location validation completed")