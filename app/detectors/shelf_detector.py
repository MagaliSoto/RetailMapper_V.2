# detectors/shelf_detector.py
from ultralytics import YOLO

class ShelfDetector:
    def __init__(self, model_path, conf_threshold=0.4):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, store, frame):
        results = self.model(frame)
        detections = []

        if not results or not hasattr(results[0], "boxes"):
            return detections  # Safe fallback if no detections

        for det in results[0].boxes:
            conf = float(det.conf[0])
            if conf < self.conf_threshold:
                continue  # Skip weak detections

            x1, y1, x2, y2 = map(int, det.xyxy[0])
            label = self.model.names[int(det.cls[0])]
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
                "store": store
            })

        return detections
