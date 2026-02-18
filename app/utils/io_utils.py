import os
import cv2
import numpy as np
from typing import List, Tuple


def load_images(input_folder: str) -> List[Tuple[str, np.ndarray]]:
    """
    Load all JPG/JPEG/PNG images from a folder.

    Parameters
    ----------
    input_folder : str
        Directory containing images.

    Returns
    -------
    List[Tuple[str, np.ndarray]]
        List of (filename, image_array) tuples in BGR format.

    Raises
    ------
    ValueError
        If input_folder is invalid.
    """

    if not input_folder or not isinstance(input_folder, str):
        raise ValueError("input_folder must be a non-empty string.")

    if not os.path.isdir(input_folder):
        raise ValueError(f"Invalid folder path: {input_folder}")

    images: List[Tuple[str, np.ndarray]] = []

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(input_folder, filename)
            img = cv2.imread(path)

            if img is not None:
                images.append((filename, img))

    return images


def save_image(
    image: np.ndarray,
    filename: str,
    output_folder: str
) -> None:
    """
    Save an image to disk, creating the output directory if needed.

    Parameters
    ----------
    image : np.ndarray
        Image in BGR format.
    filename : str
        Output filename.
    output_folder : str
        Destination directory.

    Raises
    ------
    ValueError
        If parameters are invalid.
    """

    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("image must be a valid numpy array.")

    if not filename or not isinstance(filename, str):
        raise ValueError("filename must be a non-empty string.")

    if not output_folder or not isinstance(output_folder, str):
        raise ValueError("output_folder must be a non-empty string.")

    os.makedirs(output_folder, exist_ok=True)

    out_path = os.path.join(output_folder, filename)
    cv2.imwrite(out_path, image)


def save_image_product(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    id_product: int,
    output_folder: str,
    padding_ratio: float = 0.08,
    min_size: int = 256,
    jpeg_quality: int = 98
) -> None:
    """
    Save a cropped and enhanced product image.

    Processing steps:
        1. Expand bounding box with padding.
        2. Crop image safely.
        3. Upscale if below minimum size.
        4. Apply light sharpening.
        5. Save as high-quality JPEG.

    Parameters
    ----------
    image : np.ndarray
        Original frame (BGR).
    bbox : Tuple[int, int, int, int]
        Bounding box (x, y, width, height).
    id_product : int
        Product identifier used for filename.
    output_folder : str
        Destination directory.
    padding_ratio : float
        Extra padding ratio applied to bbox.
    min_size : int
        Minimum dimension size (square threshold).
    jpeg_quality : int
        JPEG compression quality (0–100).
    """

    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("image must be a valid numpy array.")

    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        raise ValueError("bbox must be a tuple (x, y, w, h).")

    if not isinstance(output_folder, str) or not output_folder:
        raise ValueError("output_folder must be a non-empty string.")

    h_img, w_img = image.shape[:2]
    x, y, w, h = map(int, bbox)

    # ---- Expand bounding box with padding ----
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w_img, x + w + pad_x)
    y2 = min(h_img, y + h + pad_y)

    # ---- Crop region safely ----
    cropped = image[y1:y2, x1:x2]

    if cropped.size == 0:
        return  # Safety guard (invalid crop)

    # ---- Upscale small crops ----
    ch, cw = cropped.shape[:2]

    if min(ch, cw) < min_size:
        scale = min_size / min(ch, cw)
        new_w = int(cw * scale)
        new_h = int(ch * scale)

        cropped = cv2.resize(
            cropped,
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC
        )

    # ---- Apply light sharpening filter ----
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    cropped = cv2.filter2D(cropped, -1, kernel)

    # ---- Save image ----
    os.makedirs(output_folder, exist_ok=True)

    filename = f"product_{id_product}.jpg"
    out_path = os.path.join(output_folder, filename)

    cv2.imwrite(
        out_path,
        cropped,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    )


def load_image_as_numpy(image_path: str) -> np.ndarray:
    """
    Load a single image from disk as a NumPy array (BGR).

    Parameters
    ----------
    image_path : str
        Path to image file.

    Returns
    -------
    np.ndarray
        Loaded image in BGR format.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValueError
        If image cannot be decoded.
    """

    if not image_path or not isinstance(image_path, str):
        raise ValueError("image_path must be a non-empty string.")

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    return image
