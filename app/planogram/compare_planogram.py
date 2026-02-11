from collections import defaultdict
from app.utils.clip_utils import compare_images_clip
from app.config import DEBUG


# ============================================================
# Utils
# ============================================================

def build_planogram_id_to_label(label_ids_dict: dict) -> dict:
    """
    Builds a reverse lookup dictionary from planogram ID to label.
    """
    id_to_label = {}

    for label, data in label_ids_dict.items():
        for _id in data.get("ids", []):
            id_to_label[_id] = label

    return id_to_label


# ============================================================
# Main comparison
# ============================================================

def compare_planogram(
    products_detected,
    planogram_data,
    label_embeddings_dict,
    label_ids_dict,
    debug: bool = DEBUG
):
    results = {
        "match": [],
        "different_location": [],
        "unexpected": [],
        "debug": {} if debug else None,
    }

    # --------------------------------------------------
    # Build embeddings dict
    # --------------------------------------------------
    product_embeddings = {
        p["id"]: p["embedding"]
        for p in products_detected
        if "id" in p and "embedding" in p
    }

    # --------------------------------------------------
    # CLIP matching
    # --------------------------------------------------
    best_label_per_id = {}
    best_score_per_id = {}

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

    # --------------------------------------------------
    # Group matches
    # --------------------------------------------------
    matched_ids = set(best_label_per_id.keys())
    matches_by_label = defaultdict(list)

    for pid, label in best_label_per_id.items():
        matches_by_label[label].append(pid)

    all_ids = {p["id"] for p in products_detected}
    unexpected_ids = sorted(all_ids - matched_ids)

    if unexpected_ids:
        results["unexpected"].append({"null": unexpected_ids})

    # --------------------------------------------------
    # COUNT UNEXPECTED PER ROW (ONLY ROW)
    # --------------------------------------------------
    unexpected_count_by_row = defaultdict(int)

    for p in products_detected:
        if p["id"] in unexpected_ids:
            unexpected_count_by_row[p["row"]] += 1

    # --------------------------------------------------
    # Location validation
    # --------------------------------------------------
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

    return results

# ============================================================
# Build label embeddings
# ============================================================

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



