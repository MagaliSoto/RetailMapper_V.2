import cv2
from app.config import *
from .io_utils import save_image_product

def draw_shelves(frame, shelves, color=(255, 0, 0)):
    """
    Draw bottom lines and labels for shelf detections.
    shelves: list of dicts with keys ['bbox', 'label', 'conf']
    """
    for s in shelves:
        x1, y1, x2, y2 = map(int, s["bbox"])
        label = s.get("label", "shelf")
        conf = s.get("conf", 0)
        shelf_id = s.get("id", None)
        cv2.line(frame, (x1, y2), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f} | ID {shelf_id}",
                    (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def draw_products(frame, products, color=(0, 255, 0)):
    """
    Draw bounding boxes, IDs, and labels for product detections.
    products: list of dicts with keys ['bbox', 'label', 'conf', 'id']
    """
    for p in products:
        x1, y1, x2, y2 = map(int, p["bbox"])
        label = p.get("label", "product")
        conf = p.get("conf", 0)
        obj_id = p.get("id", None)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Build text label (include ID if available)
        if obj_id is not None:
            text = f"ID {obj_id} | {label} {conf:.2f}"
        else:
            text = f"{label} {conf:.2f}"

        # Save the product image
        save_image_product(frame, (x1, y1, x2 - x1, y2 - y1), obj_id, OUTPUT_FOLDER_PRODUCT)

        # Put text above the box
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame
