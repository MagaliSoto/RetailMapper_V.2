# app\planogram\compare_planogram.py

from collections import defaultdict
from typing import Dict, List, Any, Tuple
from app.utils.clip_utils import compare_images_clip_for_planogram
from app.utils.gpt_utils import call_gpt_with_images
from app.planogram.location_manager import validate_location
from app.config import GPT_SYSTEM_PROMPT, DEBUG


def _clip_matching(
    products_detected: List[Dict],
    label_embeddings_dict: Dict[str, List[Any]]
) -> Tuple[Dict[Any, str], List[Any], Dict[str, Any]]:
    """
    Perform CLIP similarity matching and collect detailed debug info.

    Returns
    -------
    - best_label_per_id : product_id → best matched label
    - unexpected_ids : list of product IDs without match
    - clip_debug : structured debug information
    """

    product_embeddings = {
        p["id"]: p["embedding"]
        for p in products_detected
        if "id" in p and "embedding" in p
    }

    best_label_per_id: Dict[Any, str] = {}
    best_score_per_id: Dict[Any, float] = {}
    threshold_per_id: Dict[Any, float] = {}

    # Debug structure
    clip_debug = {
        "per_product": {},   # id -> detailed info
        "per_label": defaultdict(list)
    }

    for label, ref_embeddings in label_embeddings_dict.items():
        for ref_embedding in ref_embeddings:

            validated, threshold, raw_sims, details = compare_images_clip_for_planogram(
                product_embeddings,
                ref_embedding,
                label
            )

            for pid, score in raw_sims.items():
                clip_debug["per_product"].setdefault(pid, {
                    "candidates": []
                })

                clip_debug["per_product"][pid]["candidates"].append({
                    "label": label,
                    **details[pid]  # ya trae score, threshold, margin, flags
                })

                is_valid = details[pid]["final_decision"]

                if not is_valid:
                    continue

                if pid not in best_score_per_id or score > best_score_per_id[pid]:
                    best_score_per_id[pid] = float(score)
                    best_label_per_id[pid] = label
                    threshold_per_id[pid] = float(threshold)

    # Construir resumen final por producto
    for pid in product_embeddings.keys():

        assigned_label = best_label_per_id.get(pid)
        best_score = best_score_per_id.get(pid)
        threshold = threshold_per_id.get(pid)

        clip_debug["per_product"][pid]["best_match"] = {
            "label": assigned_label,
            "score": float(best_score) if best_score is not None else None,
            "threshold": float(threshold) if threshold is not None else None,
            "confidence_over_threshold": (
                float(best_score - threshold)
                if best_score is not None and threshold is not None
                else None
            )
        }

        if assigned_label:
            clip_debug["per_label"][assigned_label].append({
                "id": pid,
                "score": float(best_score),
                "threshold": float(threshold),
                "margin": float(best_score - threshold)
            })

    matched_ids = set(best_label_per_id.keys())
    all_ids = {p["id"] for p in products_detected}
    unexpected_ids = sorted(all_ids - matched_ids)

    clip_debug["summary"] = {
        "total_detected": len(all_ids),
        "total_matched": len(matched_ids),
        "total_unexpected": len(unexpected_ids)
    }

    return best_label_per_id, unexpected_ids, clip_debug


def _handle_unexpected(
    unexpected_ids: List[Any],
    products_detected: List[Dict],
    results: Dict[str, Any],
    debug: bool = DEBUG
) -> Dict[Any, List[int]]:

    unexpected_cols_by_row: Dict[Any, List[Any]] = defaultdict(list)

    for pid in unexpected_ids:
        det = next(p for p in products_detected if p["id"] == pid)

        row = det["row"]
        col = det["col"]

        unexpected_cols_by_row[row].append(col)

        image_path = det.get("image_path")

        try:
            label = call_gpt_with_images(
                image_paths=[image_path],
                system_prompt=GPT_SYSTEM_PROMPT
            ) if image_path else "producto generico"

        except Exception:
            label = "producto generico"

        # 🔹 Guardar en unexpected
        results["unexpected"].append({label: [pid]})

        # 🔥 GUARDAR DEBUG PARA TASK BUILDER
        if debug:
            results.setdefault("debug", {})
            results["debug"].setdefault(str(pid), [])

            results["debug"][str(pid)].append({
                "status": "unexpected",
                "label": label,
                "detected": {
                    "row": row,
                    "adjusted_col": col
                }
            })

    return unexpected_cols_by_row

def _handle_missing(
    best_label_per_id: Dict[Any, str],
    products_detected: List[Dict],
    label_ids_dict: Dict[str, Dict[str, Any]],
    results: Dict[str, Any]
) -> Dict[Any, List[int]]:
    """
    Detect missing products from planogram.

    A product is considered missing only when the number of detected
    instances for a label is lower than the number of expected
    references in the planogram.
    """

    missing_positions_by_row = defaultdict(list)

    # Contar detectados por label
    detected_count_by_label = defaultdict(int)

    # Guardar posiciones detectadas por label
    detected_positions_by_label = defaultdict(list)

    for det in products_detected:
        pid = det["id"]
        label = best_label_per_id.get(pid)

        if label:
            detected_count_by_label[label] += 1
            detected_positions_by_label[label].append(
                (det["row"], det["col"])
            )

    for label, data in label_ids_dict.items():

        expected_rows = data.get("row", [])
        expected_cols = data.get("col", [])

        expected_positions = [
            (row, col)
            for row in expected_rows
            for col in expected_cols
        ]

        expected_total = len(expected_positions)
        detected_total = detected_count_by_label.get(label, 0)

        # 🔥 Solo hay missing si faltan unidades reales
        if detected_total >= expected_total:
            continue

        missing_quantity = expected_total - detected_total

        detected_positions = detected_positions_by_label.get(label, [])

        # Ahora sí calculamos posiciones faltantes
        missing_positions = [
            (row, col)
            for (row, col) in expected_positions
            if (row, col) not in detected_positions
        ]

        # Por seguridad: limitar a la cantidad real faltante
        missing_positions = missing_positions[:missing_quantity]

        if missing_positions:

            positions_by_row = defaultdict(list)

            for row, col in missing_positions:
                missing_positions_by_row[row].append(col)
                positions_by_row[row].append(col)

            results["missing"].append({
                "label": label,
                "missing_quantity": missing_quantity,
                "expected": expected_total,
                "detected": detected_total,
                "positions": dict(positions_by_row)
            })

    return missing_positions_by_row


def compare_planogram(
    products_detected: List[Dict],
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
        "missing": [],
        "clip_label_assignment": {},
        "clip_debug": {} if debug else None,
        "debug": {} if debug else None,
    }

    best_label_per_id, unexpected_ids, clip_debug = _clip_matching(
        products_detected,
        label_embeddings_dict
    )

    # Mantener agrupación simple
    clip_grouped = defaultdict(list)
    for pid, label in best_label_per_id.items():
        clip_grouped[label].append(pid)

    results["clip_label_assignment"] = dict(clip_grouped)

    if debug:
        results["clip_debug"] = clip_debug
    
    unexpected_cols_by_row = _handle_unexpected(
        unexpected_ids,
        products_detected,
        results,
        debug
    )

    missing_positions_by_row  = _handle_missing(
        best_label_per_id,
        products_detected,
        label_ids_dict,
        results
    )
    
    validate_location(
        best_label_per_id,
        products_detected,
        label_ids_dict,
        unexpected_cols_by_row,
        missing_positions_by_row,
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