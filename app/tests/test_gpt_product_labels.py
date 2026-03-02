"""
Test for GPT-based product label classification.

This test:
- Sends product images to GPT service
- Validates label output
- Ensures retry + fallback logic works

Requires:
- OPENAI_API_KEY configured in environment variables
"""

import os
import time

from app.utils.gpt_utils import call_gpt_with_images
from app.config import GPT_SYSTEM_PROMPT

IMAGE_FOLDER = "tmp/products"

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
import logging

# =====================================================
# 🔹 LOGGER CONFIGURATION (DEBUG)
# =====================================================

def configure_logger():
    logging.basicConfig(
        level=logging.DEBUG,  # Nivel DEBUG
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Si querés que librerías externas no spameen tanto:
    logging.getLogger("openai").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.INFO)


def main():

    configure_logger()
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Folder not found: {IMAGE_FOLDER}")
        return

    images = [
        f for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith(VALID_EXTENSIONS)
    ]

    if not images:
        print("No images found")
        return

    print(f"\nFound {len(images)} images\n")

    for img_name in images:
        img_path = os.path.join(IMAGE_FOLDER, img_name)

        try:
            label = call_gpt_with_images(
                image_paths=img_path,
                system_prompt=GPT_SYSTEM_PROMPT,
                max_retries=3
            )

            print(f"{img_name} -> {label}")

        except Exception as e:
            print(f"{img_name} -> ERROR: {e}")

        # Avoid rate limit
        time.sleep(5)


if __name__ == "__main__":
    main()
