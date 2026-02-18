# app\planogram\compare_planogram.py

from collections import defaultdict
from typing import Dict, List, Any, Tuple
from app.utils.clip_utils import compare_images_clip
from app.utils.gpt_utils import call_gpt_with_images
from app.config import GPT_SYSTEM_PROMPT, DEBUG


def _clip_matching(
    products_detected: List[Dict],
    label_embeddings_dict: Dict[str, List[Any]]
) -> Tuple[Dict[Any, str], List[Any]]:
    """
    Perform CLIP similarity matching between detected products
    and reference label embeddings.

    Parameters
    ----------
    products_detected : List[Dict]
        Detected products with 'id' and 'embedding'.
    label_embeddings_dict : Dict[str, List[Any]]
        Precomputed embeddings per label.

    Returns
    -------
    Tuple[Dict[Any, str], List[Any]]
        - best_label_per_id : product_id → best matched label
        - unexpected_ids : list of product IDs without match
    """
    product_embeddings = {
        p["id"]: p["embedding"]
        for p in products_detected
        if "id" in p and "embedding" in p
    }

    best_label_per_id: Dict[Any, str] = {}
    best_score_per_id: Dict[Any, float] = {}

    for label, ref_embeddings in label_embeddings_dict.items():
        for ref_embedding in ref_embeddings:
            sims, threshold = compare_images_clip(
                product_embeddings,
                ref_embedding,
                label
            )

            for pid, score in sims.items():
                if score < threshold:
                    continue

                if pid not in best_score_per_id or score > best_score_per_id[pid]:
                    best_score_per_id[pid] = score
                    best_label_per_id[pid] = label

    matched_ids = set(best_label_per_id.keys())
    all_ids = {p["id"] for p in products_detected}
    unexpected_ids = sorted(all_ids - matched_ids)

    return best_label_per_id, unexpected_ids


def _handle_unexpected(
    unexpected_ids: List[Any],
    products_detected: List[Dict],
    results: Dict[str, Any]
) -> Dict[Any, int]:
    """
    Handle unexpected products using GPT classification
    and count unexpected items per row.

    Parameters
    ----------
    unexpected_ids : List[Any]
        Product IDs without CLIP match.
    products_detected : List[Dict]
        Full detected products list.
    results : Dict[str, Any]
        Results dictionary to append unexpected items.

    Returns
    -------
    Dict[Any, int]
        Count of unexpected products per row.
    """
    unexpected_count_by_row: Dict[Any, int] = defaultdict(int)

    for pid in unexpected_ids:
        det = next(p for p in products_detected if p["id"] == pid)

        unexpected_count_by_row[det["row"]] += 1

        image_path = det.get("image_path")
        if not image_path:
            continue

        try:
            label = call_gpt_with_images(
                image_paths=[image_path],
                system_prompt=GPT_SYSTEM_PROMPT
            )
            
        except Exception:
            label = "producto generico"

        results["unexpected"].append({label: [pid]})

    return unexpected_count_by_row


def _validate_location(
    best_label_per_id: Dict[Any, str],
    products_detected: List[Dict],
    label_ids_dict: Dict[str, Dict[str, Any]],
    unexpected_count_by_row: Dict[Any, int],
    results: Dict[str, Any],
    debug: bool
) -> None:
    """
    Validate detected product locations against expected planogram positions.

    Parameters
    ----------
    best_label_per_id : Dict[Any, str]
        Product ID → matched label.
    products_detected : List[Dict]
        Detected products with row/col/subrow.
    label_ids_dict : Dict[str, Dict[str, Any]]
        Expected planogram metadata per label.
    unexpected_count_by_row : Dict[Any, int]
        Count of unexpected products per row.
    results : Dict[str, Any]
        Results dictionary to update.
    debug : bool
        If True, store detailed debug information.
    """
    matches_by_label: Dict[str, List[Any]] = defaultdict(list)

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
            adjusted_col = det["col"] - extras

            status = (
                det["row"] in expected_rows and
                adjusted_col in expected_cols and
                det["subrow"] in expected_subrows
            )

            if status:
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
                    "status": "match" if status else "different_location"
                })

        if ok_ids:
            results["match"].append({label: ok_ids})
        if wrong_ids:
            results["different_location"].append({label: wrong_ids})


def compare_planogram(
    products_detected: List[Dict],
    planogram_data: List[Dict],
    label_embeddings_dict: Dict[str, List[Any]],
    label_ids_dict: Dict[str, Dict[str, Any]],
    debug: bool = DEBUG
) -> Dict[str, Any]:
    """
    Compare detected products against planogram reference.

    Orchestrates:
    - CLIP matching
    - Unexpected handling
    - Location validation

    Returns structured comparison results.
    """
    results = {
        "match": [],
        "different_location": [],
        "unexpected": [],
        "debug": {} if debug else None,
    }

    best_label_per_id, unexpected_ids = _clip_matching(
        products_detected,
        label_embeddings_dict
    )

    unexpected_count_by_row = _handle_unexpected(
        unexpected_ids,
        products_detected,
        results
    )

    _validate_location(
        best_label_per_id,
        products_detected,
        label_ids_dict,
        unexpected_count_by_row,
        results,
        debug
    )

    return results

def build_label_embeddings_from_planogram(label_ids_dict, planogram_data):
    planogram_embeddings_by_id = {
        p["id"]: p["embedding"]
        for p in planogram_data
        if "id" in p and "embedding" in p
    }

    label_embeddings = {}

    for label, data in label_ids_dict.items():
        embeddings = [
            planogram_embeddings_by_id[pid]
            for pid in data.get("ids", [])
            if pid in planogram_embeddings_by_id
        ]

        if embeddings:
            label_embeddings[label] = embeddings

    return label_embeddings