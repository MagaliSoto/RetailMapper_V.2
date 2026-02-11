"""
Script de prueba para el pipeline de procesamiento de imágenes.
Ejecutar desde la raíz del proyecto.

Ejemplo:
python test_process_image_pipeline.py
"""

import os, json
from pprint import pprint

from app.services.planogram_pipeline import process_image_pipeline
from app.utils.json_utils import save_products_to_json

def main():
    # ---- Configuración manual (alternativa si no querés usar config) ----
    # img_path = "data/test_images/shelf_01.jpg"
    # n_shelf = 1
    # id_store = "store_001"

    img_path = "input_images_test\imgGondola2.jpeg"
    n_shelf = 1
    id_store = 112

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"No existe la imagen: {img_path}")

    print("🚀 Iniciando pipeline")
    print(f"📸 Imagen: {img_path}")
    print(f"🧱 Shelf: {n_shelf} | 🏬 Store: {id_store}")

    _, dict = process_image_pipeline(
        img_path=img_path,
        n_shelf=n_shelf,
        id_store=id_store
    )
    
    OUTPUT_JSON = "final_data_test.json"
    
    os.makedirs(os.path.dirname(OUTPUT_JSON) or ".", exist_ok=True)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            dict,
            f,
            ensure_ascii=False,
            indent=4
        )
    
if __name__ == "__main__":
    main()
