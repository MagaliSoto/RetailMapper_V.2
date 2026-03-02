import os
import cv2

from app.detectors.product_detector import ProductDetector
from app.utils.draw import draw_products  # ajustá si cambia el path


def main():
    # ---------------- CONFIG ----------------
    model_path = "models\product-model.pt"          # <-- tu modelo
    input_image_path = "input_images_test\img1.jpeg"      # <-- imagen de prueba
    crops_output_dir = "tmp/products"
    debug_output_dir = "tmp/debug"

    os.makedirs(debug_output_dir, exist_ok=True)

    # ---------------- LOAD IMAGE ----------------
    frame = cv2.imread(input_image_path)
    if frame is None:
        raise ValueError(f"No se pudo cargar la imagen: {input_image_path}")

    # ---------------- INIT DETECTOR ----------------
    detector = ProductDetector(
        model_path=model_path,
        conf_threshold=0.3
    )

    # ---------------- RUN DETECTION ----------------
    products = detector.detect(
        shelf=1,
        frame=frame,
        output_dir=crops_output_dir
    )

    print(f"Detecciones encontradas: {len(products)}")

    # ---------------- DRAW BBOXES ----------------
    debug_frame = frame.copy()
    debug_frame = draw_products(debug_frame, products)

    # ---------------- SAVE DEBUG IMAGE ----------------
    output_path = os.path.join(debug_output_dir, "detections_debug.jpg")
    cv2.imwrite(output_path, debug_frame)

    print(f"Imagen con bounding boxes guardada en: {output_path}")


if __name__ == "__main__":
    main()