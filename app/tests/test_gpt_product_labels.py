import os, time
from app.utils.gpt_utils import call_gpt_with_images
from app.config import GPT_SYSTEM_PROMPT

IMAGE_FOLDER = "output_products"

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def main():
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

    for img_name in images:
        img_path = os.path.join(IMAGE_FOLDER, img_name)

        try:
            label = call_gpt_with_images(
                image_path=img_path,
                user_prompt=GPT_SYSTEM_PROMPT,
                max_retries=3
            )

            print(f"{img_name} -> {label}")


        except Exception as e:
            print(f"{img_name} -> ERROR: {e}")
        
        time.sleep(10)


if __name__ == "__main__":
    main()
