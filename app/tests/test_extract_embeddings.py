"""
Test for CLIP embedding extraction logic.

This script validates:
- Product crop extraction
- Embedding generation using CLIP
- Correct attachment of embedding vectors to product objects

Purpose:
Ensure embedding pipeline works independently of full audit pipeline.
"""

import json
from app.utils.clip_utils import extract_product_embeddings
from app.utils.io_utils import load_image_as_numpy


IMAGE_PATH = "input_images_test/imgGondola2.jpeg"
PRODUCTS_JSON = "output/products.json"


def main():
    """
    Loads detected products and image, then generates embeddings.
    """

    image = load_image_as_numpy(IMAGE_PATH)

    with open(PRODUCTS_JSON, "r", encoding="utf-8") as f:
        products = json.load(f)

    print("Extracting embeddings...\n")

    extract_product_embeddings(products, image)

    print("Embedding extraction completed.")

    if "embedding" in products[0]:
        print("✅ Embeddings successfully attached to products.")
    else:
        print("❌ Embedding field missing.")


if __name__ == "__main__":
    main()
