import os
import json
import logging
import numpy as np
from typing import Any, Dict, List, Union


logger = logging.getLogger(__name__)


def save_groups_to_json(
    data_groups: Dict[str, List[int]],
    filename: str,
    output_folder: str
) -> None:
    """
    Save grouped classification results into a structured JSON file.

    Output structure:
        {
            "total_groups": int,
            "total_products": int,
            "groups": [
                {
                    "label": str,
                    "product_ids": List[int],
                    "count": int
                }
            ]
        }

    Parameters
    ----------
    data_groups : Dict[str, List[int]]
        Mapping of label → list of product IDs.
    filename : str
        Base filename (extension replaced with "_groups.json").
    output_folder : str
        Destination directory.

    Raises
    ------
    ValueError
        If parameters are invalid.
    """

    if not isinstance(data_groups, dict):
        raise ValueError("data_groups must be a dictionary.")

    if not filename or not isinstance(filename, str):
        raise ValueError("filename must be a non-empty string.")

    if not output_folder or not isinstance(output_folder, str):
        raise ValueError("output_folder must be a non-empty string.")

    os.makedirs(output_folder, exist_ok=True)

    total_products = sum(len(ids) for ids in data_groups.values())

    structured_output = {
        "total_groups": len(data_groups),
        "total_products": total_products,
        "groups": []
    }

    for label, ids in data_groups.items():
        structured_output["groups"].append({
            "label": label,
            "product_ids": ids,
            "count": len(ids)
        })

    json_filename = os.path.splitext(filename)[0] + "_groups.json"
    json_path = os.path.join(output_folder, json_filename)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(structured_output, f, indent=4, ensure_ascii=False)

    logger.info(
        "Groups JSON saved",
        extra={"path": json_path}
    )


def save_products_to_json(
    products: List[Dict[str, Any]],
    filename: str,
    output_folder: str
) -> None:
    """
    Save product dictionaries to JSON.

    All values are converted to JSON-safe format.
    Non-serializable values are stored as None.

    Parameters
    ----------
    products : List[Dict[str, Any]]
        List of product dictionaries.
    filename : str
        Output filename (without extension).
    output_folder : str
        Destination directory.
    """

    if not isinstance(products, list):
        raise ValueError("products must be a list.")

    if not filename or not isinstance(filename, str):
        raise ValueError("filename must be a non-empty string.")

    if not output_folder or not isinstance(output_folder, str):
        raise ValueError("output_folder must be a non-empty string.")

    os.makedirs(output_folder, exist_ok=True)

    data: List[Dict[str, Any]] = []

    for product in products:
        if not isinstance(product, dict):
            continue

        product_json = {
            key: to_json_safe(value)
            for key, value in product.items()
        }

        data.append(product_json)

    json_filename = os.path.splitext(filename)[0] + ".json"
    json_path = os.path.join(output_folder, json_filename)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    logger.info(
        "Products JSON saved",
        extra={"path": json_path}
    )


def to_json_safe(value: Any) -> Any:
    """
    Convert a value into a JSON-serializable format.

    Supports:
        - torch.Tensor → list
        - np.ndarray → list
        - Primitive types (int, float, str, bool, list, tuple, dict)

    Parameters
    ----------
    value : Any
        Value to convert.

    Returns
    -------
    Any
        JSON-safe representation or None if unsupported.
    """

    if value is None:
        return None

    # torch tensor
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except ImportError:
        pass

    # numpy array
    if isinstance(value, np.ndarray):
        return value.tolist()

    # primitive types
    if isinstance(value, (int, float, str, bool, list, tuple, dict)):
        return value

    # fallback
    return None


def parse_json_to_list(
    json_data: Union[List[Dict[str, Any]], str]
) -> List[Dict[str, Any]]:
    """
    Convert JSON input into a flat list of dictionaries.

    If a string is provided, it is treated as a file path.

    Parameters
    ----------
    json_data : Union[List[Dict[str, Any]], str]
        List of dictionaries or path to JSON file.

    Returns
    -------
    List[Dict[str, Any]]
        Flat list of dictionaries.
    """

    if isinstance(json_data, str):
        if not os.path.isfile(json_data):
            raise FileNotFoundError(f"JSON file not found: {json_data}")

        with open(json_data, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json_data

    if not isinstance(data, list):
        raise ValueError("JSON data must be a list of dictionaries.")

    result_list: List[Dict[str, Any]] = []

    for item in data:
        if isinstance(item, dict):
            result_list.append({k: v for k, v in item.items()})

    return result_list


def load_planogram_from_json(path: str) -> List[Dict[str, Any]]:
    """
    Load planogram data from JSON and convert embeddings to numpy arrays.

    Expected JSON structure:
        [
            {
                "id": int,
                "row": int,
                "subrow": int,
                "col": int,
                "embedding": List[float]
            }
        ]

    Parameters
    ----------
    path : str
        Path to JSON file.

    Returns
    -------
    List[Dict[str, Any]]
        Planogram data with embeddings as np.ndarray(dtype=float32).
    """

    if not path or not isinstance(path, str):
        raise ValueError("path must be a non-empty string.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Planogram JSON must contain a list.")

    planogram_data: List[Dict[str, Any]] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        if "id" not in item or "embedding" not in item:
            continue

        planogram_data.append({
            "id": item["id"],
            "row": item.get("row"),
            "subrow": item.get("subrow"),
            "col": item.get("col"),
            "embedding": np.array(
                item["embedding"],
                dtype="float32"
            )
        })

    logger.info(
        "Planogram loaded",
        extra={"num_items": len(planogram_data)}
    )

    return planogram_data
