# app/core/model_loader.py

import os
import shutil
from threading import Lock
from app.detectors.shelf_detector import ShelfDetector
from app.detectors.product_detector import ProductDetector
from app.utils.hf_utils import download_model


_shelf_detector = None
_product_detector = None
_lock = Lock()


def _ensure_model_in_local_folder(
    cached_path: str,
    local_path: str
) -> str:
    """
    Ensure the model file exists in the target local path.
    If missing, copy it from the HuggingFace cache location.

    Parameters
    ----------
    cached_path : str
        Path returned by HuggingFace cache download.
    local_path : str
        Desired local destination path for the model.

    Returns
    -------
    str
        Final local path where the model is stored.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if not os.path.exists(local_path):
        shutil.copyfile(cached_path, local_path)

    return local_path


def load_models() -> tuple[ShelfDetector, ProductDetector]:
    """
    Load shelf and product detection models (singleton pattern).
    Models are downloaded if necessary and initialized once.

    Returns
    -------
    tuple[ShelfDetector, ProductDetector]
        Initialized shelf and product detector instances.
    """
    global _shelf_detector, _product_detector

    with _lock:

        if _shelf_detector is None:
            cached_shelf_path = download_model(
                repo_id="MagaliSoto/RetailMapper",
                filename="shelf-model.pt",
                local_path="models/shelf-model.pt"
            )

            final_shelf_path = _ensure_model_in_local_folder(
                cached_shelf_path,
                "models/shelf-model.pt"
            )

            _shelf_detector = ShelfDetector(final_shelf_path)

        if _product_detector is None:
            cached_product_path = download_model(
                repo_id="MagaliSoto/RetailMapper",
                filename="product-model.pt",
                local_path="models/product-model.pt"
            )

            final_product_path = _ensure_model_in_local_folder(
                cached_product_path,
                "models/product-model.pt"
            )

            _product_detector = ProductDetector(final_product_path)

    return _shelf_detector, _product_detector


def get_models() -> tuple[ShelfDetector, ProductDetector]:
    """
    Retrieve previously loaded model instances.

    Returns
    -------
    tuple[ShelfDetector, ProductDetector]
        Shelf and product detector instances.

    Raises
    ------
    RuntimeError
        If models were not loaded before calling this function.
    """
    if _shelf_detector is None or _product_detector is None:
        raise RuntimeError("Models not loaded. Did startup run?")

    return _shelf_detector, _product_detector
