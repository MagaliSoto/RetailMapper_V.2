import os
import json
import numpy as np

def save_groups_to_json(
    data_groups: dict,
    filename: str,
    output_folder: str
):
    """
    Saves grouped classification results to JSON.

    Args:
        data_groups (Dict[str, List[int]]):
            Mapping of label -> list of product IDs.
        filename (str):
            Base filename (extension will be replaced with .json).
        output_folder (str):
            Folder where JSON file will be stored.

    The output structure includes basic metadata:
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
    """

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

    print(f"✅ Groups JSON saved: {json_path}")


def save_products_to_json(products, filename, output_folder):
    """
    Save products dynamically to JSON.
    Any missing or non-serializable field is stored as None.
    """
    os.makedirs(output_folder, exist_ok=True)

    data = []

    for p in products:
        product_json = {}

        for key, value in p.items():
            product_json[key] = to_json_safe(value)

        data.append(product_json)

    json_filename = os.path.splitext(filename)[0] + ".json"
    json_path = os.path.join(output_folder, json_filename)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"✅ JSON saved: {json_path}")

def to_json_safe(value):
    if value is None:
        return None

    # torch tensor (PRIMERO)
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except ImportError:
        pass

    # numpy array
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass

    # tipos simples
    if isinstance(value, (int, float, str, bool, list, tuple, dict)):
        return value

    # fallback
    return None


def parse_json_to_list(json_data):
    """
    Convierte un JSON (lista de diccionarios) en una lista plana de diccionarios
    asegurando que cada elemento tenga todas las claves disponibles.

    Args:
        json_data (list[dict] o str): lista de diccionarios o ruta a un archivo JSON.

    Returns:
        list[dict]: lista de diccionarios lista para buscar por claves.
    """
    # Si json_data es un string, asumimos que es una ruta a archivo JSON
    if isinstance(json_data, str):
        with open(json_data, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json_data  # asumimos que ya es lista de diccionarios

    result_list = []
    for item in data:
        # Copiamos todas las claves existentes en cada diccionario
        result_list.append({k: v for k, v in item.items()})

    return result_list

def load_planogram_from_json(path: str):
    """
    Loads planogram_data from JSON and converts embeddings to numpy arrays.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    planogram_data = []

    for item in data:
        if "id" not in item or "embedding" not in item:
            continue

        planogram_data.append({
            "id": item["id"],
            "row": item.get("row"),
            "subrow": item.get("subrow"),
            "col": item.get("col"),
            "embedding": np.array(item["embedding"], dtype="float32")
        })

    return planogram_data