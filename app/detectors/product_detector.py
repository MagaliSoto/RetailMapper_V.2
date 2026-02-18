# app/detectors/product_detector.py

import os
import cv2
from typing import Union
from ultralytics import YOLO


class ProductDetector:
    """
    YOLO-based product detector.
    Performs inference over a frame and saves cropped product images.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.4,
        output_dir: str = "tmp/products"
    ) -> None:
        """
        Initialize the product detector.

        Parameters
        ----------
        model_path : str
            Path to the YOLO model weights.
        conf_threshold : float
            Minimum confidence required to keep a detection.
        output_dir : str
            Directory where cropped product images will be stored.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    def detect(
        self,
        shelf: Union[int, str],
        frame
    ) -> list[dict]:
        """
        Run product detection on a frame and save cropped images.

        Parameters
        ----------
        shelf : Union[int, str]
            Shelf identifier associated with this detection pass.
        frame : numpy.ndarray
            Image frame in BGR format.

        Returns
        -------
        list[dict]
            List of detected products with:
            - 'bbox' (tuple[int, int, int, int])
            - 'conf' (float)
            - 'shelf' (Union[int, str])
            - 'image_path' (str)
        """
        results = self.model(frame)
        detections = []

        if not results or not hasattr(results[0], "boxes"):
            return detections

        product_id = 0

        for det in results[0].boxes:
            conf = float(det.conf[0])
            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, det.xyxy[0])

            # Crop product region
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            product_id += 1
            img_path = os.path.join(
                self.output_dir,
                f"shelf_{shelf}_product_{product_id}.jpg"
            )

            cv2.imwrite(img_path, crop)

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
                "shelf": int(shelf),
                "image_path": img_path
            })

        return detections
