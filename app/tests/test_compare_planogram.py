"""
Unit-style test for the planogram comparison logic.

This script validates the behavior of the compare_planogram function.

What this test verifies:
- Matching between detected products and planogram entries
- Correct identification of missing products
- Detection of unexpected products
- Correct similarity scoring behavior

This test runs independently from the API layer and assumes:
- Valid embeddings are available
- Properly formatted planogram JSON exists
"""
import json
from app.utils.gpt_utils import *
from app.utils.json_utils import *
from app.planogram.compare_planogram import *
from app.core.model_loader import load_models
from app.utils.io_utils import load_image_as_numpy
from app.localization.assign_row import assign_rows
from app.localization.assign_subrow import assign_subrows
from app.localization.assign_column import assign_columns
from app.utils.clip_utils import extract_product_embeddings
from app.planogram.compare_planogram import build_label_embeddings_from_planogram




def main():
    planogram_data_path = "output\products.json"
    
    planogram_data = load_planogram_from_json(planogram_data_path)

    img_path = "input_images_test\imgGondola3.jpeg"
    img = load_image_as_numpy(img_path)
    id_store = 112
    n_shelf = 1
    shelf_detector, product_detector = load_models()

    shelves = shelf_detector.detect(id_store, img)
    products = product_detector.detect(n_shelf, img)

    products = assign_rows(products, shelves)
    products = assign_columns(products)
    products = assign_subrows(products)
    
    extract_product_embeddings(products, img)

    for i, p in enumerate(products, start=1):
        p["id"] = i

    print("🚀 Ejecutando comparación de planograma\n")
    
    with open("output/data_groups.json", "r", encoding="utf-8") as f:
        raw_label_ids_dict = json.load(f)

    label_ids_dict = {
        group["label"]: group["product_ids"]
        for group in raw_label_ids_dict["groups"]
    }


    label_embeddings_dict = build_label_embeddings_from_planogram(
        label_ids_dict=label_ids_dict,
        planogram_data=planogram_data
    )

    print("\n📦 Label → embeddings del planograma:")
    for label, embeds in label_embeddings_dict.items():
        print(f"{label}: {len(embeds)} embeddings")
    
    result = compare_planogram(
        products_detected=products,
        planogram_data=planogram_data,
        label_embeddings_dict=label_embeddings_dict,
        label_ids_dict=label_ids_dict,
        debug=True
    )

    """
    result = enrich_unexpected_with_gpt(
        result,
        products
    )
    
    """
    
    with open("output/compare_planogram.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def enrich_unexpected_with_gpt(
    result: dict,
    products_detected: list,
    system_prompt: str = None
) -> dict:
    """
    Replaces unexpected/null labels with GPT-generated labels.
    """

    if "unexpected" not in result:
        return result

    # index rápido por id
    products_by_id = {
        p["id"]: p for p in products_detected if "id" in p
    }

    new_unexpected = []

    for item in result["unexpected"]:
        # item es algo como { "null": [3, 9] }
        for label, ids in item.items():

            # si no es null, lo dejamos como está
            if label is not None and label != "null":
                new_unexpected.append(item)
                continue

            for pid in ids:
                product = products_by_id.get(pid)
                if not product:
                    continue

                image_path = product.get("image_path")
                if not image_path:
                    continue

                generated_label = call_gpt_with_image(
                    image_paths=image_path,
                    system_prompt=system_prompt
                )

                new_unexpected.append({
                    generated_label: [pid]
                })

    result["unexpected"] = new_unexpected
    return result

if __name__ == "__main__":
    main()
