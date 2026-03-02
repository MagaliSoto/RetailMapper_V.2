"""
Unified CLIP embedding and grouping module.

Responsibilities
----------------
- Load CLIP model (singleton pattern)
- Convert images to normalized embeddings
- Compute cosine similarity
- Apply adaptive similarity thresholding
- Perform internal similarity grouping
- Provide CLIP-based product clustering utilities

IMPORTANT:
This module does NOT configure logging globally.
Logging configuration must be handled by the application entrypoint.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Union, Optional, Set
from collections import defaultdict
from PIL import Image
from io import BytesIO
import open_clip


logger = logging.getLogger(__name__)


# ============================================================
# Device configuration
# ============================================================

device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# CLIP model singleton loader
# ============================================================

_clip_model: Optional[torch.nn.Module] = None
_preprocess = None


def get_clip() -> Tuple[torch.nn.Module, Any]:
    """
    Load CLIP ViT-H-14 model once and reuse it.

    Returns
    -------
    Tuple[torch.nn.Module, Any]
        Loaded model and preprocess transform.
    """
    global _clip_model, _preprocess

    if _clip_model is None:
        logger.info("Loading CLIP model", extra={"model": "ViT-H-14"})
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

def ensure_tensor(
    x: Any,
    unsqueeze: bool = False
) -> torch.Tensor:
    """
    Ensure input is a torch.Tensor located on the correct device.

    Parameters
    ----------
    x : Any
        Input embedding-like object.
    unsqueeze : bool
        If True and input is 1D, adds batch dimension.

    Returns
    -------
    torch.Tensor
        Tensor on target device.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    x = x.to(device)

    if unsqueeze and x.ndim == 1:
        x = x.unsqueeze(0)

    return x


def safe_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Perform safe L2 normalization.

    Prevents division by zero by adding epsilon.

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    torch.Tensor
        L2-normalized tensor.
    """
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)


# ============================================================
# Image → embedding
# ============================================================

def image_to_embedding(
    image: Union[np.ndarray, bytes, bytearray, Image.Image]
) -> Optional[torch.Tensor]:
    """
    Convert image input into normalized CLIP embedding.

    Supported formats:
    - numpy array (OpenCV BGR expected)
    - raw bytes
    - PIL Image

    Parameters
    ----------
    image : Union[np.ndarray, bytes, bytearray, PIL.Image.Image]

    Returns
    -------
    Optional[torch.Tensor]
        Normalized embedding vector [D] or None.
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
# Cosine similarity
# ============================================================

def cosine_similarity(
    a: Union[torch.Tensor, np.ndarray, List[float]],
    b: Union[torch.Tensor, np.ndarray, List[float]]
) -> float:
    """
    Compute cosine similarity between two embeddings.

    Parameters
    ----------
    a : embedding-like
    b : embedding-like

    Returns
    -------
    float
        Cosine similarity score.
    """
    a = safe_normalize(ensure_tensor(a))
    b = safe_normalize(ensure_tensor(b))

    return float(torch.dot(a.flatten(), b.flatten()))


# ============================================================
# Similarity computation
# ============================================================

def compute_similarities(
    ref_embedding: Any,
    embeddings: Dict[int, Any]
) -> Dict[int, float]:
    """
    Compute similarity between reference embedding and group.

    Parameters
    ----------
    ref_embedding : Any
        Reference embedding.
    embeddings : Dict[int, Any]
        id → embedding.

    Returns
    -------
    Dict[int, float]
        id → similarity score.
    """
    ref_embedding = safe_normalize(
        ensure_tensor(ref_embedding, unsqueeze=True)
    )

    results: Dict[int, float] = {}

    for id_, emb in embeddings.items():
        if emb is None:
            continue

        emb = safe_normalize(
            ensure_tensor(emb, unsqueeze=True)
        )

        sim = torch.matmul(ref_embedding, emb.T).item()
        results[id_] = float(sim)

    return results


# ============================================================
# Adaptive threshold
# ============================================================

def determine_matches(
    similarities: Dict[int, float],
    label_ref: str = "",
    method: str = "dynamic",
    top_margin: float = 0.2,
    tolerance: float = 0.02
) -> Tuple[Dict[int, bool], float]:
    """
    Determine which similarities pass adaptive threshold.

    Strategy:
    - Compute top similarity
    - Adjust threshold dynamically based on std deviation
    - Apply tolerance correction

    Returns
    -------
    Tuple[Dict[int, bool], float]
        Match map and threshold value.
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

    logger.info(
        "Adaptive threshold computed",
        extra={
            "label": label_ref,
            "threshold": float(threshold),
            "top_similarity": float(top_sim),
            "std": float(std_val)
        }
    )

    matches = {
        id_num: sim >= threshold
        for id_num, sim in similarities.items()
    }

    return matches, threshold


# ============================================================
# Public comparison interface
# ============================================================

def compare_images_clip(
    detected_embeddings: Dict[int, Any],
    ref_embedding: Any,
    label_ref: str = "",
    use_text: bool = False,
    return_scores: bool = True,
    logs: bool = True,
    method: str = "dynamic",
    top_margin: float = 0.2,
    tolerance: float = 0.03
) -> Tuple[Dict[int, float], float]:
    """
    Compare reference embedding against detected embeddings.

    ALWAYS returns (dict, threshold)
    """
    if ref_embedding is None:
        logger.error("Reference embedding is None")
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
        logger.info(
            "CLIP comparison result",
            extra={
                "label": label_ref,
                "threshold": float(threshold),
                "num_candidates": len(sim_visual)
            }
        )

    if return_scores:
        return sim_visual, threshold

    return matches, threshold


# ============================================================
# Internal similarity grouping
# ============================================================

def expand_group_by_internal_similarity(
    seed_ids: List[int],
    embeddings: Dict[int, Any],
    min_sim: float = 0.80
) -> Set[int]:
    """
    Expand group iteratively based on internal similarity graph.
    """
    expanded: Set[int] = set(seed_ids)
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
    Split a group into internally coherent subgroups.
    """
    subgroups: List[List[int]] = []
    used: Set[int] = set()

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
# Main clustering entrypoint
# ============================================================

def classify_reference(
    product_embeddings: Dict[int, Any],
    internal_sim_threshold: float = 0.88,
    debug: bool = False
) -> Dict[int, List[int]]:
    """
    Cluster visually similar products using CLIP similarity.
    """
    groups: Dict[int, List[int]] = {}
    used: Set[int] = set()
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

    logger.info(
        "CLIP clustering completed",
        extra={"num_groups": len(groups)}
    )

    return groups

def crop_from_bbox(
    image: np.ndarray,
    bbox: List[int]
) -> np.ndarray:
    """
    Crop a region from an image using a bounding box.

    Parameters
    ----------
    image : np.ndarray
        Full image in OpenCV format (BGR expected).
    bbox : List[int]
        Bounding box coordinates in format [x1, y1, x2, y2].

    Returns
    -------
    np.ndarray
        Cropped image region.

    Notes
    -----
    Coordinates are cast to integers before slicing.
    No bounds checking is performed (assumes valid bbox).
    """
    x1, y1, x2, y2 = map(int, bbox)
    return image[y1:y2, x1:x2]


def compute_clip_embedding(
    img_path: str
) -> Optional[torch.Tensor]:
    """
    Compute a normalized CLIP embedding from an image file path.

    This function preserves the original public contract used
    across the application.

    Parameters
    ----------
    img_path : str
        Path to image file.

    Returns
    -------
    Optional[torch.Tensor]
        Normalized embedding tensor [D] or None if an error occurs.

    Logging
    -------
    Emits structured error logs if image loading fails.
    """
    try:
        image = Image.open(img_path).convert("RGB")
        return image_to_embedding(image)

    except Exception as e:
        logger.error(
            "Failed to compute CLIP embedding",
            extra={
                "image_path": img_path,
                "error": str(e)
            }
        )
        return None


def fuse_clip_embeddings(
    embeddings: List[torch.Tensor],
    normalize: bool = True
) -> Optional[torch.Tensor]:
    """
    Fuse multiple CLIP embeddings into a single representative embedding.

    Strategy
    --------
    - Filter invalid embeddings
    - Mean pooling across embeddings
    - Optional L2 normalization

    Intended Use
    ------------
    Useful for reference items (e.g. planogram products)
    where multiple sample images represent one product.

    Parameters
    ----------
    embeddings : List[torch.Tensor]
        List of embedding tensors [D] or [1, D].
    normalize : bool
        Whether to L2 normalize the fused embedding.

    Returns
    -------
    Optional[torch.Tensor]
        Fused embedding tensor [D] or None if no valid inputs.
    """

    if not embeddings:
        return None

    valid_embeddings: List[torch.Tensor] = []

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

    logger.debug(
        "Embeddings fused",
        extra={"num_embeddings": len(valid_embeddings)}
    )

    return fused


def extract_product_embeddings(
    products: List[Dict[str, Any]],
    image: np.ndarray
) -> None:
    """
    Add CLIP embeddings to products in-place using batch GPU processing.

    This implementation preserves the original public contract:
        - Same parameters
        - Same return type (None)
        - Each product gets:
            product["embedding"] = torch.Tensor or None

    Internally, it performs batch inference for GPU efficiency.
    """

    if not products:
        return

    model, preprocess = get_clip()

    valid_products = []
    processed_tensors = []

    # Collect valid crops and preprocess them
    for product in products:
        bbox = product.get("bbox")

        if bbox is None:
            product["embedding"] = None
            continue

        crop = crop_from_bbox(image, bbox)

        try:
            pil_img = Image.fromarray(crop[..., ::-1]).convert("RGB")
            tensor = preprocess(pil_img)
            processed_tensors.append(tensor)
            valid_products.append(product)
        except Exception as e:
            logger.warning(
                "Failed preprocessing crop",
                extra={"error": str(e)}
            )
            product["embedding"] = None

    # If no valid crops → exit safely
    if not processed_tensors:
        return

    # Batch forward pass (REAL GPU batching)
    batch = torch.stack(processed_tensors).to(device)

    with torch.no_grad():
        embeddings = model.encode_image(batch)
        embeddings = safe_normalize(embeddings)

    # Assign embeddings back in original order
    for product, emb in zip(valid_products, embeddings):
        product["embedding"] = emb.cpu()


def compare_images_clip_for_planogram(
    product_embeddings: Dict[Any, Any],
    reference_embedding: Any,
    label: str = "",
    min_absolute_score: float = 0.30,
    min_margin_over_threshold: float = 0.10,
    debug: bool = False,
):
    """
    Specialized CLIP comparison for inter-image planogram matching.

    Adds stricter validation rules on top of the dynamic threshold logic.

    Returns:
        - validated_matches: Dict[id, score]
        - threshold: float
        - raw_scores: Dict[id, float]
        - validation_details: Dict[id, Dict]
    """

    sims, threshold = compare_images_clip(
        product_embeddings,
        reference_embedding,
        label
    )

    validated_matches = {}
    validation_details = {}

    for pid, score in sims.items():

        passes_dynamic = score >= threshold
        passes_absolute = score >= min_absolute_score
        margin = score - threshold
        passes_margin = margin >= min_margin_over_threshold

        is_valid = (
            passes_dynamic
            and passes_absolute
            and passes_margin
        )

        validation_details[pid] = {
            "score": float(score),
            "threshold": float(threshold),
            "margin_over_threshold": float(margin),
            "passes_dynamic": bool(passes_dynamic),
            "passes_absolute": bool(passes_absolute),
            "passes_margin": bool(passes_margin),
            "final_decision": bool(is_valid),
        }

        if is_valid:
            validated_matches[pid] = float(score)

    return validated_matches, threshold, sims, validation_details