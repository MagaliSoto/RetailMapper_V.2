import os
import time
import logging
from app.config import *
from typing import Dict, List
from collections import defaultdict
from app.utils.io_utils import load_image_as_numpy
from app.localization.assign_row import assign_rows
from app.utils.gpt_utils import call_gpt_with_images
from app.localization.assign_column import assign_columns
from app.localization.assign_subrow import assign_subrows
from app.core.model_loader import get_models, load_models 
from app.utils.clip_utils import (
    extract_product_embeddings,
    classify_reference,
    compare_images_clip
)

logger = logging.getLogger(__name__)


# Main Pipeline

def process_image_pipeline(img_path, n_shelf, id_store):
    """
    Full image processing pipeline:
    1. Load image
    2. Detect shelves and products
    3. Assign spatial metadata (row/column/subrow)
    4. Extract embeddings
    5. Cluster products and classify with GPT

    Returns:
        products (List[dict])
        data_groups (Dict[label, List[product_ids]])
    """

    pipeline_start = time.perf_counter()
    logger.info(f"Starting pipeline for image: {img_path}")

    data_groups = {}

    # Image Loading
    t0 = time.perf_counter()
    img = load_image_as_numpy(img_path)
    logger.info(f"Image loaded in {time.perf_counter() - t0:.3f}s")

    # Model Loading (should be singleton in production)
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

    # Assign sequential IDs
    for i, p in enumerate(products, start=1):
        p["id"] = i

    # Embedding extraction
    t0 = time.perf_counter()
    extract_product_embeddings(products, img)
    logger.info(f"Embeddings extracted in {time.perf_counter() - t0:.3f}s")

    # Group classification
    if products:
        t0 = time.perf_counter()
        data_groups = classify_groups(products)
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

def build_embedding_and_path_maps(products):
    """
    Convert product list into:
    - id -> embedding map
    - id -> image_path map

    This avoids repeated lookups and improves clarity.
    """
    embedding_map = {}
    path_map = {}

    for product in products:
        pid = product["id"]
        embedding_map[pid] = product.get("embedding")
        path_map[pid] = product.get("image_path")

    return embedding_map, path_map


# Group Classification

def classify_groups(
    products: List[dict],
    strict_clip_threshold: float = 0.88,
    internal_sim_threshold: float = 0.70,
    debug: bool = False,
) -> Dict[str, List[int]]:
    """
    Classifies detected products into semantic groups using:

    1. CLIP clustering
    2. GPT validation
    3. Strict CLIP fallback refinement

    Returns:
        Dict[label, List[product_ids]]
    """

    start_time = time.perf_counter()
    final_labels = defaultdict(list)

    product_embeddings, id_to_path = build_embedding_and_path_maps(products)

    # Initial CLIP grouping
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
            logger.info(f"Accepted label '{label}' for group {gid}")
            continue

        logger.warning(f"GPT returned ERROR for group {gid}. Applying fallback.")

        # Strict CLIP refinement
        refined_groups = _refine_group_with_strict_clip(
            ids,
            product_embeddings,
            strict_clip_threshold
        )

        logger.info(
            f"Refined into {len(refined_groups)} subgroups for group {gid}"
        )

        for sub_ids in refined_groups:

            sub_paths = _extract_image_paths(sub_ids, id_to_path)

            t0 = time.perf_counter()
            label = _ask_gpt_group_label(sub_paths)
            logger.info(
                f"GPT response for refined subgroup in "
                f"{time.perf_counter() - t0:.3f}s"
            )

            if label == "ERROR":
                logger.warning(f"Subgroup {sub_ids} discarded (still ERROR)")
                continue

            final_labels[label].extend(sub_ids)
            logger.info(f"Refined label '{label}' → {sub_ids}")

    logger.info(
        f"Classification finished in {time.perf_counter() - start_time:.3f}s"
    )

    return dict(final_labels)


# GPT & CLIP Utilities

def _extract_image_paths(ids, id_to_path):
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
    Sends a group of product images to GPT and asks for a single label.
    Returns:
        label (str) or "ERROR"
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
        user_prompt=user_prompt,
        system_prompt=GPT_SYSTEM_PROMPT
    )

    return label.strip() if label else "ERROR"


def _refine_group_with_strict_clip(
    ids: List[int],
    embeddings: Dict[int, any],
    threshold: float
) -> List[List[int]]:
    """
    Refines a group using a stricter CLIP similarity threshold.
    Helps separate visually similar but distinct products.
    """

    refined_groups = []
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
