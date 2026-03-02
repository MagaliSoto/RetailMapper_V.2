"""
Direct test for the process_image_pipeline function.

This script validates:
- Shelf detection
- Product detection
- Spatial grouping
- JSON-ready output formatting

Purpose:
Test detection + grouping pipeline independently from API.
"""

import json
import time

from app.services.process_planogram_pipeline import process_image_pipeline
from app.utils.json_utils import save_products_to_json, save_groups_to_json
from app.core.model_loader import load_models
from app.config import OUTPUT_FOLDER
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

IMAGE_PATH = "input_images_test/img1.jpeg"
N_SHELF = 1
ID_STORE = 2


def main():
    """
    Executes full process pipeline locally.
    """

    print("Loading models...")
    load_models()

    print("Running process_image_pipeline...\n")

    start_time = time.perf_counter()

    products, groups = process_image_pipeline(
        IMAGE_PATH,
        N_SHELF,
        ID_STORE
    )

    total_time = time.perf_counter() - start_time

    print(f"Pipeline completed in {total_time:.3f}s")
    print(f"Detected products: {len(products)}")
    print(f"Generated groups: {len(groups)}")

    save_products_to_json(
                products=products,
                filename="products.json",
                output_folder=OUTPUT_FOLDER
            )
    
    save_groups_to_json(
        data_groups=groups,
        filename="data.json",
        output_folder=OUTPUT_FOLDER
    )



if __name__ == "__main__":
    main()
