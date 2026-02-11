# detectors/product_detector.py
import os
import cv2
from ultralytics import YOLO


class ProductDetector:
    def __init__(
        self,
        model_path,
        conf_threshold=0.4,
        output_dir="tmp/products"
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    def detect(self, shelf, frame):
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

            # Crop product
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
                "shelf": shelf,
                "image_path": img_path
            })

        return detections
