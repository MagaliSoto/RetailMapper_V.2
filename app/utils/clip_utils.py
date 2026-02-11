"""
Unified CLIP embedding and grouping module.

This module provides:
- CLIP model loading (singleton)
- Image → embedding conversion
- Embedding fusion
- Cosine similarity computation
- Adaptive threshold matching
- Internal similarity grouping
- Graph-based clustering utilities
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Any
from collections import defaultdict
from PIL import Image
from io import BytesIO
import open_clip


# ============================================================
# Device configuration
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)


# ============================================================
# CLIP model singleton loader
# ============================================================

_clip_model = None
_preprocess = None


def get_clip():
    """
    Loads CLIP ViT-H-14 once and keeps it in memory.
    """
    global _clip_model, _preprocess

    if _clip_model is None:
        logging.info("Loading CLIP ViT-H-14 model...")
        _clip_model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14",
            pretrained="laion2b_s32b_b79k"
        )
        _clip_model = _clip_model.to(device)
        _clip_model.eval()

    return _clip_model, _preprocess


# ============================================================
# Tensor utilities
# ============================================================

def ensure_tensor(x: Any, unsqueeze: bool = False) -> torch.Tensor:
    """
    Ensures input is a torch.Tensor on the correct device.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    x = x.to(device)

    if unsqueeze and x.ndim == 1:
        x = x.unsqueeze(0)

    return x


def safe_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    L2 normalizes a tensor safely.
    """
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)


# ============================================================
# Image → embedding
# ============================================================

def image_to_embedding(image) -> torch.Tensor | None:
    """
    Converts numpy / bytes / PIL image into normalized CLIP embedding.
    """
    if image is None:
        return None

    model, preprocess = get_clip()

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image[..., ::-1]).convert("RGB")

    elif isinstance(image, (bytes, bytearray)):
        image = Image.open(BytesIO(image)).convert("RGB")

    elif isinstance(image, Image.Image):
        image = image.convert("RGB")

    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_image(tensor)
        emb = safe_normalize(emb)

    return emb.squeeze(0)


# ============================================================
# Cosine similarity (unified)
# ============================================================

def cosine_similarity(a, b) -> float:
    """
    Computes cosine similarity between two embeddings.
    Accepts torch / numpy / list.
    """
    a = ensure_tensor(a)
    b = ensure_tensor(b)

    a = safe_normalize(a)
    b = safe_normalize(b)

    return float(torch.dot(a.flatten(), b.flatten()))


# ============================================================
# Similarity computation against group
# ============================================================

def compute_similarities(
    ref_embedding: Any,
    embeddings: Dict[int, Any]
) -> Dict[int, float]:
    """
    Computes cosine similarity between reference and a group.
    """
    ref_embedding = ensure_tensor(ref_embedding, unsqueeze=True)
    ref_embedding = safe_normalize(ref_embedding)

    results = {}

    for id_, emb in embeddings.items():
        if emb is None:
            continue

        emb = ensure_tensor(emb, unsqueeze=True)
        emb = safe_normalize(emb)

        sim = torch.matmul(ref_embedding, emb.T).item()
        results[id_] = float(sim)

    return results


# ============================================================
# Adaptive threshold
# ============================================================

def determine_matches(
    similarities: dict,
    label_ref: str = "",
    method: str = "dynamic",
    top_margin: float = 0.2,
    tolerance: float = 0.02
) -> tuple[dict, float]:
    """
    Determines which similarities pass an adaptive threshold.
    Preserves original interface from Code 1.

    Returns:
        (matches_dict, threshold)
    """
    if not similarities:
        return {}, 0.0

    sims = np.array(list(similarities.values()))
    top_id, top_sim = max(similarities.items(), key=lambda x: x[1])
    std_val = sims.std()

    if method == "fixed":
        threshold = top_sim - top_margin
    else:
        threshold = max(top_sim - std_val * 0.5, top_sim - top_margin)

    dynamic_tolerance = max(tolerance, top_sim * 0.025 + std_val * 0.25)
    threshold -= dynamic_tolerance

    logging.info(
        f"Threshold for '{label_ref}': {threshold:.3f} "
        f"(tol={dynamic_tolerance:.3f}, std={std_val:.3f})"
    )

    matches = {id_num: sim >= threshold for id_num, sim in similarities.items()}

    return matches, threshold


# ============================================================
# Main comparison function (unified)
# ============================================================

def compare_images_clip(
    detected_embeddings: dict,
    ref_embedding,
    label_ref: str = "",
    use_text: bool = False,
    return_scores: bool = True,
    logs: bool = True,
    method: str = "dynamic",
    top_margin: float = 0.2,
    tolerance: float = 0.03
):
    """
    Original public interface for CLIP comparison.
    ALWAYS returns (dict, threshold)
    """

    if ref_embedding is None:
        logging.error("Reference embedding is None")
        return {}, 0.0

    sim_visual = compute_similarities(ref_embedding, detected_embeddings)

    if not sim_visual:
        return {}, 0.0

    matches, threshold = determine_matches(
        sim_visual,
        label_ref=label_ref,
        method=method,
        top_margin=top_margin,
        tolerance=tolerance
    )

    if logs:
        print(f"\n[Comparing label '{label_ref}']")
        print(f"Threshold: {threshold:.3f}")
        print(f"{'ID':<6} | {'Similarity':<10} | Match")
        print("-" * 40)

        for id_num, sim in sorted(sim_visual.items(), key=lambda x: x[1], reverse=True):
            is_match = matches.get(id_num, False)
            mark = "✔️" if is_match else "❌"
            print(f"{id_num:<6} | {sim:<10.3f} | {mark}")

    if return_scores:
        return sim_visual, threshold

    return matches, threshold



# ============================================================
# Internal coherence grouping
# ============================================================

def expand_group_by_internal_similarity(
    seed_ids: List[int],
    embeddings: Dict[int, Any],
    min_sim: float = 0.80
) -> set[int]:
    """
    Expands a group based on internal similarity.
    """
    expanded = set(seed_ids)
    changed = True

    while changed:
        changed = False
        for i in list(expanded):
            for j, emb_j in embeddings.items():
                if j in expanded:
                    continue
                if cosine_similarity(embeddings[i], emb_j) >= min_sim:
                    expanded.add(j)
                    changed = True

    return expanded


def split_by_internal_similarity(
    ids: List[int],
    embeddings: Dict[int, Any],
    min_sim: float = 0.88
) -> List[List[int]]:
    """
    Splits a group into coherent subgroups.
    """
    subgroups = []
    used = set()

    for i in ids:
        if i in used:
            continue

        group = [i]
        used.add(i)

        for j in ids:
            if j in used:
                continue

            if cosine_similarity(embeddings[i], embeddings[j]) >= min_sim:
                group.append(j)
                used.add(j)

        subgroups.append(group)

    return subgroups


# ============================================================
# Main classification
# ============================================================

def classify_reference(
    product_embeddings: Dict[int, Any],
    internal_sim_threshold: float = 0.88,
    debug: bool = False
) -> Dict[int, List[int]]:
    """
    Groups visually similar products.
    """

    groups = {}
    used = set()
    group_id = 1

    for ref_id, ref_emb in product_embeddings.items():

        if ref_id in used:
            continue

        sims, threshold = compare_images_clip(
            product_embeddings,
            ref_emb,
            label_ref=f"ref_{ref_id}"
        )

        matched = [k for k, v in sims.items() if v >= threshold]

        expanded = expand_group_by_internal_similarity(
            matched,
            product_embeddings,
            min_sim=0.70
        )

        subgroups = split_by_internal_similarity(
            list(expanded),
            product_embeddings,
            min_sim=internal_sim_threshold
        )

        for sg in subgroups:
            groups[group_id] = sorted(sg)
            used.update(sg)
            group_id += 1

    return groups

def crop_from_bbox(image: np.ndarray, bbox: list[int]) -> np.ndarray:
    """
    Crops an image using a bounding box [x1, y1, x2, y2].

    Args:
        image (np.ndarray): Full image (BGR expected if from OpenCV)
        bbox (list[int]): Bounding box coordinates [x1, y1, x2, y2]

    Returns:
        np.ndarray: Cropped image region
    """
    x1, y1, x2, y2 = map(int, bbox)
    return image[y1:y2, x1:x2]


def compute_clip_embedding(img_path: str) -> torch.Tensor | None:
    """
    Computes a CLIP embedding from an image file path.

    IMPORTANT:
    This function preserves the original contract used in other modules.

    Args:
        img_path (str): Path to image file

    Returns:
        torch.Tensor | None: Normalized embedding or None if error
    """
    try:
        image = Image.open(img_path).convert("RGB")
        return image_to_embedding(image)
    except Exception as e:
        logging.error(f"Failed to compute embedding for {img_path}: {e}")
        return None


def extract_product_embeddings(products: list, image: np.ndarray):
    """
    Adds a CLIP embedding to each product dictionary in-place.

    Each product dictionary must contain:
        {
            "bbox": [x1, y1, x2, y2]
        }

    The function adds:
        product["embedding"] = torch.Tensor or None

    Args:
        products (list): List of product dicts
        image (np.ndarray): Full image (OpenCV format expected)
    """
    for product in products:
        bbox = product.get("bbox")

        if bbox is None:
            product["embedding"] = None
            continue

        crop = crop_from_bbox(image, bbox)
        emb = image_to_embedding(crop)

        product["embedding"] = emb.cpu() if emb is not None else None


def fuse_clip_embeddings(
    embeddings: list[torch.Tensor],
    normalize: bool = True
) -> torch.Tensor | None:
    """
    Fuses multiple CLIP embeddings into a single representative embedding
    using mean pooling.

    Intended for reference products (e.g. planogram items).

    Args:
        embeddings (List[torch.Tensor]): List of embeddings [D] or [1, D]
        normalize (bool): Whether to L2 normalize final result

    Returns:
        torch.Tensor | None
    """
    if not embeddings:
        return None

    valid_embeddings = []

    for emb in embeddings:
        if emb is None:
            continue

        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb, dtype=torch.float32)

        if emb.ndim == 2:
            emb = emb.squeeze(0)

        valid_embeddings.append(emb)

    if not valid_embeddings:
        return None

    stacked = torch.stack(valid_embeddings, dim=0)
    fused = stacked.mean(dim=0)

    if normalize:
        fused = fused / (fused.norm() + 1e-8)

    return fused
