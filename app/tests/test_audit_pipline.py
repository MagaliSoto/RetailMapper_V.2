"""
Direct unit-style test for audit_image_pipeline (without API).

This script:
- Loads models manually
- Loads planogram JSON
- Runs the full audit pipeline
- Measures execution time
- Optionally saves output to file

Purpose:
Validate audit logic independently from FastAPI.
"""

import json
import time
import os

from app.services.audit_pipline import audit_image_pipeline
from app.core.model_loader import load_models
import logging

logging.basicConfig(
    level=logging.INFO,  # DEBUG si querés todo
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

IMAGE_PATH = "input_images_test/img4.jpeg"
LABEL_IDS_JSON_PATH = "output/data_groups.json"
PLANOGRAM_JSON_PATH = "output/products.json"

N_SHELF = 1
ID_STORE = 2

SAVE_OUTPUT = True
OUTPUT_PATH = "output/tasks.json"


def main():
    """
    Executes the audit pipeline locally and prints summary.
    """

    print("\nLoading models...")
    load_models()

    with open(LABEL_IDS_JSON_PATH, "r", encoding="utf-8") as f:
        label_ids_dict = json.load(f)

    with open(PLANOGRAM_JSON_PATH, "r", encoding="utf-8") as f:
        planogram_data = json.load(f)

    print("Starting audit pipeline test...\n")

    start_time = time.perf_counter()

    products_comp = audit_image_pipeline(
        n_shelf=N_SHELF,
        id_store=ID_STORE,
        img_path=IMAGE_PATH,
        label_ids_dict=label_ids_dict,
        planogram_data=planogram_data
    )

    total_time = time.perf_counter() - start_time

    print(f"\nPipeline finished in {total_time:.3f}s")
    print(f"Total compared products: {len(products_comp)}")

    print("\nSample output:")
    print(json.dumps(products_comp, indent=2, ensure_ascii=False))

    if SAVE_OUTPUT:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(products_comp, f, indent=4, ensure_ascii=False)
        print(f"\nOutput saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
