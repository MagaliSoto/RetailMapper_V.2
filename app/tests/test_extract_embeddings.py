import cv2
import numpy as np
from pprint import pprint

from utils.clip_utils import extract_product_embeddings


def main():
    # -----------------------------
    # Cargar imagen de prueba
    # -----------------------------
    image_path = "output\Imagen.jpg"  # <-- cambia esto
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

    print(f"[INFO] Imagen cargada: {image.shape}")

    # -----------------------------
    # Productos de prueba
    # -----------------------------
    products = [
        {
            "id": 1,
            "bbox": [50, 50, 200, 200]
        },
        {
            "id": 2,
            "bbox": [220, 60, 380, 240]
        },
        {
            "id": 3,
            "bbox": None  # caso borde
        }
    ]

    # -----------------------------
    # Extraer embeddings
    # -----------------------------
    extract_product_embeddings(products, image)

    # -----------------------------
    # Mostrar resultados
    # -----------------------------
    print("\n[RESULTADOS]")
    for p in products:
        emb = p.get("embedding")

        print(f"\nProducto ID: {p['id']}")
        if emb is None:
            print("  ❌ Embedding: None")
        else:
            print(f"  ✅ Embedding generado")
            print(f"  Dimensión: {len(emb)}")
            print(f"  Primeros 5 valores: {emb[:5]}")


if __name__ == "__main__":
    main()
