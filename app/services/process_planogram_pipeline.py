import os
import time
import logging
from typing import Dict, List, Tuple, Any
from collections import defaultdict

from app.config import *
from app.utils.io_utils import load_image_as_numpy
from app.localization.assign_row import assign_rows
from app.localization.assign_column import assign_columns
from app.localization.assign_subrow import assign_subrows
from app.utils.gpt_utils import call_gpt_with_images
from app.config import GPT_SYSTEM_PROMPT
from app.core.model_loader import get_models
from app.utils.clip_utils import (
    extract_product_embeddings,
    classify_reference,
    compare_images_clip
)

logger = logging.getLogger(__name__)


def process_image_pipeline(
    img_path: str,
    n_shelf: Any,
    id_store: Any
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """
    Execute full image processing pipeline.

    Steps:
    - Load image
    - Detect shelves and products
    - Assign spatial metadata
    - Extract embeddings
    - Cluster and classify products

    Parameters
    ----------
    img_path : str
        Path to input image.
    n_shelf : Any
        Shelf identifier for product detection.
    id_store : Any
        Store identifier for shelf detection.

    Returns
    -------
    Tuple[List[Dict], Dict[str, Dict]]
        - products : detected products with metadata
        - data_groups : structured classification groups
    """

    if not isinstance(img_path, str):
        raise TypeError("img_path must be a string")

    pipeline_start = time.perf_counter()
    logger.info(f"Starting pipeline for image: {img_path}")

    data_groups: Dict[str, Dict] = {}

    # Image loading
    t0 = time.perf_counter()
    img = load_image_as_numpy(img_path)
    logger.info(f"Image loaded in {time.perf_counter() - t0:.3f}s")

    # Model retrieval (singleton expected)
    t0 = time.perf_counter()
    shelf_detector, product_detector = get_models()
    logger.info(f"Models retrieved in {time.perf_counter() - t0:.3f}s")

    # Detection
    t0 = time.perf_counter()
    shelves = shelf_detector.detect(id_store, img)
    products = product_detector.detect(n_shelf, img)
    logger.info(
        f"Detection completed in {time.perf_counter() - t0:.3f}s | "
        f"Shelves: {len(shelves)} | Products: {len(products)}"
    )

    # Spatial assignment
    t0 = time.perf_counter()
    products = assign_rows(products, shelves)
    products = assign_columns(products)
    products = assign_subrows(products)
    logger.info(f"Spatial assignment completed in {time.perf_counter() - t0:.3f}s")

    # Assign sequential product IDs
    for i, p in enumerate(products, start=1):
        p["id"] = i

    # Embedding extraction
    t0 = time.perf_counter()
    extract_product_embeddings(products, img)
    logger.info(f"Embeddings extracted in {time.perf_counter() - t0:.3f}s")

    # Group classification
    if products:
        t0 = time.perf_counter()

        raw_groups = classify_groups(products)

        data_groups = build_structured_groups(
            products=products,
            grouped_ids=raw_groups
        )

        logger.info(
            f"Group classification completed in {time.perf_counter() - t0:.3f}s | "
            f"Groups found: {len(data_groups)}"
        )
    else:
        logger.warning("No products detected. Skipping classification.")

    logger.info(
        f"Pipeline finished in {time.perf_counter() - pipeline_start:.3f}s | "
        f"Total products: {len(products)}"
    )

    return products, data_groups


# Helper Utilities

def build_embedding_and_path_maps(
    products: List[Dict]
) -> Tuple[Dict[int, Any], Dict[int, str]]:
    """
    Build lookup maps for embeddings and image paths.

    Parameters
    ----------
    products : List[Dict]
        Detected products with 'id', 'embedding', and 'image_path'.

    Returns
    -------
    Tuple[Dict[int, Any], Dict[int, str]]
        - id → embedding
        - id → image_path
    """

    embedding_map: Dict[int, Any] = {}
    path_map: Dict[int, str] = {}

    for product in products:
        pid = product["id"]
        embedding_map[pid] = product.get("embedding")
        path_map[pid] = product.get("image_path")

    return embedding_map, path_map


# Group Classification

def classify_groups(
    products: List[Dict],
    strict_clip_threshold: float = 0.88,
    internal_sim_threshold: float = 0.70,
    debug: bool = False,
) -> Dict[str, List[int]]:
    """
    Classify detected products into semantic groups.

    Strategy:
    1. CLIP clustering
    2. GPT validation
    3. Strict CLIP refinement fallback

    Parameters
    ----------
    products : List[Dict]
        Products with 'id', 'embedding', and 'image_path'.
    strict_clip_threshold : float
        Threshold for strict fallback CLIP similarity.
    internal_sim_threshold : float
        Threshold for initial clustering.
    debug : bool
        If True, enables debug behavior inside CLIP.

    Returns
    -------
    Dict[str, List[int]]
        Mapping label → list of product IDs.
    """

    start_time = time.perf_counter()
    final_labels: Dict[str, List[int]] = defaultdict(list)

    product_embeddings, id_to_path = build_embedding_and_path_maps(products)

    groups = classify_reference(
        product_embeddings,
        internal_sim_threshold=internal_sim_threshold,
        debug=debug
    )

    logger.info(f"Initial CLIP groups: {len(groups)}")

    for gid, ids in groups.items():

        logger.info(f"Processing group {gid} with {len(ids)} products")

        image_paths = _extract_image_paths(ids, id_to_path)

        t0 = time.perf_counter()
        label = _ask_gpt_group_label(image_paths)
        logger.info(
            f"GPT response for group {gid} in {time.perf_counter() - t0:.3f}s"
        )

        if label != "ERROR":
            final_labels[label].extend(ids)
            continue

        logger.warning(f"GPT returned ERROR for group {gid}. Applying fallback.")

        refined_groups = _refine_group_with_strict_clip(
            ids,
            product_embeddings,
            strict_clip_threshold
        )

        for sub_ids in refined_groups:

            sub_paths = _extract_image_paths(sub_ids, id_to_path)

            label = _ask_gpt_group_label(sub_paths)

            if label == "ERROR":
                continue

            final_labels[label].extend(sub_ids)

    logger.info(
        f"Classification finished in {time.perf_counter() - start_time:.3f}s"
    )

    return dict(final_labels)


# GPT & CLIP Utilities

def _extract_image_paths(
    ids: List[int],
    id_to_path: Dict[int, str]
) -> List[str]:
    """
    Extract valid image paths for given product IDs.
    """
    return [
        id_to_path[pid]
        for pid in ids
        if pid in id_to_path and id_to_path[pid]
    ]

def _ask_gpt_group_label(image_paths: List[str]) -> str:
    """
    Send product images to GPT and request a single strict label.
    Returns label or "ERROR".
    """

    user_prompt = (
        "Las siguientes imágenes muestran productos detectados en una góndola.\n"
        "Todas las imágenes corresponden SUPUESTAMENTE al mismo producto.\n\n"
        "Si los productos que ves en TODAS las imágenes son el MISMO producto:\n"
        "- devolvé un único label siguiendo estrictamente las reglas.\n\n"
        "Si ves que NO corresponden al mismo producto:\n"
        '- devolvé exactamente la palabra: ERROR'
    )

    label = call_gpt_with_images(
        image_paths=image_paths,
        system_prompt=GPT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    return label



def _refine_group_with_strict_clip(
    ids: List[int],
    embeddings: Dict[int, Any],
    threshold: float
) -> List[List[int]]:
    """
    Refine a group using stricter CLIP similarity threshold.

    Parameters
    ----------
    ids : List[int]
        Product IDs belonging to the group.
    embeddings : Dict[int, Any]
        id → embedding map.
    threshold : float
        Similarity threshold.

    Returns
    -------
    List[List[int]]
        List of refined subgroups.
    """

    refined_groups: List[List[int]] = []
    used = set()

    for ref_id in ids:

        if ref_id in used:
            continue

        ref_emb = embeddings[ref_id]

        sim_scores, _ = compare_images_clip(
            {i: embeddings[i] for i in ids},
            ref_emb
        )

        refined = [
            i for i, score in sim_scores.items()
            if score >= threshold
        ]

        if refined:
            refined_groups.append(refined)
            used.update(refined)

    return refined_groups


def build_structured_groups(
    products: List[Dict],
    grouped_ids: Dict[str, List[int]]
) -> Dict[str, Dict[str, List[int]]]:
    """
    Build structured output including spatial metadata.

    Parameters
    ----------
    products : List[Dict]
        Detected products with spatial metadata.
    grouped_ids : Dict[str, List[int]]
        label → product IDs.

    Returns
    -------
    Dict[str, Dict[str, List[int]]]
        label → {
            "ids": List[int],
            "row": List[int],
            "col": List[int],
            "subrow": List[int]
        }
    """

    id_to_product = {p["id"]: p for p in products}

    structured_output: Dict[str, Dict[str, List[int]]] = {}

    for label, ids in grouped_ids.items():

        rows = set()
        cols = set()
        subrows = set()

        for pid in ids:
            product = id_to_product.get(pid)
            if not product:
                continue

            if "row" in product:
                rows.add(product["row"])

            if "col" in product:
                cols.add(product["col"])

            if "subrow" in product:
                subrows.add(product["subrow"])

        structured_output[label] = {
            "ids": ids,
            "row": sorted(rows),
            "col": sorted(cols),
            "subrow": sorted(subrows)
        }

    return structured_output
