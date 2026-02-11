import os
import cv2
import numpy as np

def load_images(input_folder):
    """
    Load all .jpg/.jpeg/.png images from the given folder.
    Returns a list of tuples: (filename, image)
    """
    images = []
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(input_folder, filename)
            img = cv2.imread(path)
            if img is not None:
                images.append((filename, img))
    return images

def save_image(image, filename, output_folder):
    """
    Save an image to the output folder, creating it if necessary.
    """
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, filename)
    cv2.imwrite(out_path, image)

def save_image_product(
    image,
    bbox,
    id_product,
    output_folder,
    padding_ratio=0.08,
    min_size=256,
    jpeg_quality=98
):
    """
    Saves a high-quality cropped image of a detected product.

    Args:
        image (np.ndarray): Original frame (BGR).
        bbox (tuple): (x, y, w, h).
        id_product (int): Product ID.
        output_folder (str): Destination folder.
        padding_ratio (float): Extra padding around bbox.
        min_size (int): Minimum output size (square).
        jpeg_quality (int): JPEG quality (0–100).
    """
    h_img, w_img = image.shape[:2]
    x, y, w, h = bbox

    # --- Expand bbox (padding) ---
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w_img, x + w + pad_x)
    y2 = min(h_img, y + h + pad_y)

    # --- Crop ---
    cropped = image[y1:y2, x1:x2]

    if cropped.size == 0:
        return  # safety guard

    # --- Upscale if too small ---
    ch, cw = cropped.shape[:2]
    if min(ch, cw) < min_size:
        scale = min_size / min(ch, cw)
        new_w = int(cw * scale)
        new_h = int(ch * scale)
        cropped = cv2.resize(
            cropped, (new_w, new_h),
            interpolation=cv2.INTER_CUBIC
        )

    # --- Optional: light sharpening ---
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    cropped = cv2.filter2D(cropped, -1, kernel)

    # --- Save ---
    os.makedirs(output_folder, exist_ok=True)
    filename = f"product_{id_product}.jpg"
    out_path = os.path.join(output_folder, filename)

    cv2.imwrite(
        out_path,
        cropped,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    )

    
def load_image_as_numpy(image_path):
    """
    Load a single image from disk and return it as a NumPy array (BGR).
    Raises an error if the image cannot be loaded.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    return image