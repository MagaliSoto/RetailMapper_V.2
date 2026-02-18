# app/detectors/shelf_detector.py

from typing import Union
from ultralytics import YOLO


class ShelfDetector:
    """
    YOLO-based shelf detector.
    Performs inference over a frame and returns shelf bounding boxes.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.4
    ) -> None:
        """
        Initialize the shelf detector.

        Parameters
        ----------
        model_path : str
            Path to the YOLO model weights.
        conf_threshold : float
            Minimum confidence required to keep a detection.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(
        self,
        store: Union[int, str],
        frame
    ) -> list[dict]:
        """
        Run shelf detection on a frame.

        Parameters
        ----------
        store : Union[int, str]
            Store identifier associated with this detection pass.
        frame : numpy.ndarray
            Image frame in BGR format.

        Returns
        -------
        list[dict]
            List of detected shelves with:
            - 'bbox' (tuple[int, int, int, int])
            - 'conf' (float)
            - 'store' (Union[int, str])
        """
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
