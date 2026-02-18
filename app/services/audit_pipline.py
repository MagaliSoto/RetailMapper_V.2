import time
import logging
from typing import Dict, List, Any, Union, Tuple

from app.config import *
from app.utils.io_utils import load_image_as_numpy
from app.localization.assign_row import assign_rows
from app.localization.assign_column import assign_columns
from app.localization.assign_subrow import assign_subrows
from app.core.model_loader import get_models
from app.utils.clip_utils import extract_product_embeddings
from app.planogram.compare_planogram import (
    compare_planogram,
    build_label_embeddings_from_planogram
)

logger = logging.getLogger(__name__)


def audit_image_pipeline(
    img_path: str,
    n_shelf: Union[int, str],
    id_store: Union[int, str],
    label_ids_dict: Dict[str, Any],
    planogram_data: List[Dict]
) -> Dict[str, Any]:
    """
    Execute full audit pipeline for a single image.

    Steps:
    - Load image
    - Detect shelves and products
    - Assign spatial metadata
    - Extract embeddings
    - Compare against planogram

    Returns
    -------
    Dict[str, Any]
        Planogram comparison results.
    """

    _validate_parameters(img_path, label_ids_dict, planogram_data)

    pipeline_start = time.perf_counter()
    logger.info(f"Starting audit pipeline for image: {img_path}")

    # Image loading
    img = _load_image_stage(img_path)

    # Model retrieval
    shelf_detector, product_detector = _load_models_stage()

    # Detection
    shelves, products = _detection_stage(
        shelf_detector,
        product_detector,
        id_store,
        n_shelf,
        img
    )

    # Spatial assignment
    products = _localization_stage(products, shelves)

    # Assign sequential IDs
    _assign_product_ids(products)

    # Embedding extraction
    _embedding_stage(products, img)

    # Planogram comparison
    results = _comparison_stage(
        products,
        label_ids_dict,
        planogram_data
    )

    logger.info(
        f"Audit pipeline finished in "
        f"{time.perf_counter() - pipeline_start:.3f}s | "
        f"Products detected: {len(products)}"
    )

    return results


# ============================================================
# Validation
# ============================================================

def _validate_parameters(
    img_path: str,
    label_ids_dict: Dict[str, Any],
    planogram_data: List[Dict]
) -> None:
    """
    Validate input parameters.
    """

    if not isinstance(img_path, str):
        raise TypeError("img_path must be a string")

    if not isinstance(label_ids_dict, dict):
        raise TypeError("label_ids_dict must be a dictionary")

    if not isinstance(planogram_data, list):
        raise TypeError("planogram_data must be a list")


# ============================================================
# Pipeline Stages
# ============================================================

def _load_image_stage(img_path: str):
    t0 = time.perf_counter()
    img = load_image_as_numpy(img_path)
    logger.info(f"Image loaded in {time.perf_counter() - t0:.3f}s")
    return img


def _load_models_stage():
    t0 = time.perf_counter()
    shelf_detector, product_detector = get_models()
    logger.info(f"Models retrieved in {time.perf_counter() - t0:.3f}s")
    return shelf_detector, product_detector


def _detection_stage(
    shelf_detector,
    product_detector,
    id_store,
    n_shelf,
    img
) -> Tuple[List[Dict], List[Dict]]:
    t0 = time.perf_counter()

    shelves = shelf_detector.detect(id_store, img)
    products = product_detector.detect(n_shelf, img)

    logger.info(
        f"Detection completed in {time.perf_counter() - t0:.3f}s | "
        f"Shelves: {len(shelves)} | Products: {len(products)}"
    )

    return shelves, products


def _localization_stage(
    products: List[Dict],
    shelves: List[Dict]
) -> List[Dict]:
    t0 = time.perf_counter()

    products = assign_rows(products, shelves)
    products = assign_columns(products)
    products = assign_subrows(products)

    logger.info(
        f"Spatial assignment completed in "
        f"{time.perf_counter() - t0:.3f}s"
    )

    return products


def _assign_product_ids(products: List[Dict]) -> None:
    """
    Assign sequential IDs to detected products.
    """
    for i, p in enumerate(products, start=1):
        p["id"] = i


def _embedding_stage(
    products: List[Dict],
    img
) -> None:
    t0 = time.perf_counter()
    extract_product_embeddings(products, img)
    logger.info(
        f"Embeddings extracted in "
        f"{time.perf_counter() - t0:.3f}s"
    )


def _comparison_stage(
    products: List[Dict],
    raw_label_ids_dict: Dict[str, Any],
    planogram_data: List[Dict]
) -> Dict[str, Any]:
    """
    Execute planogram comparison stage.

    This function:
    - Normalizes label_ids schema (from data_groups.json)
    - Builds label embeddings
    - Executes planogram comparison

    It is schema-robust and independent from external JSON structure.
    """

    # --------------------------------------------------
    # 1️⃣ Validate input schema
    # --------------------------------------------------
    if "groups" not in raw_label_ids_dict:
        raise ValueError("Invalid label_ids_dict: missing 'groups' key")

    # --------------------------------------------------
    # 2️⃣ Normalize schema
    # Convert:
    # {
    #   "groups": [
    #       { "label": "...", "product_ids": {...} }
    #   ]
    # }
    # → 
    # {
    #   "label": { ids, row, col, subrow }
    # }
    # --------------------------------------------------
    label_ids_dict = {}

    for group in raw_label_ids_dict["groups"]:
        label = group.get("label")
        product_data = group.get("product_ids")

        if not label or not isinstance(product_data, dict):
            continue

        label_ids_dict[label] = {
            "ids": product_data.get("ids", []),
            "row": product_data.get("row", []),
            "col": product_data.get("col", []),
            "subrow": product_data.get("subrow", []),
        }

    if not label_ids_dict:
        raise ValueError("No valid label groups found after normalization")

    # --------------------------------------------------
    # 3️⃣ Build embeddings dictionary
    # --------------------------------------------------
    label_embeddings_dict = build_label_embeddings_from_planogram(
        label_ids_dict=label_ids_dict,
        planogram_data=planogram_data
    )

    if not label_embeddings_dict:
        raise ValueError(
            "label_embeddings_dict is empty. "
            "Check that planogram_data contains embeddings."
        )

    # --------------------------------------------------
    # 4️⃣ Execute comparison
    # --------------------------------------------------
    return compare_planogram(
        products_detected=products,
        planogram_data=planogram_data,
        label_embeddings_dict=label_embeddings_dict,
        label_ids_dict=label_ids_dict
    )
