from app.localization.assign_row import assign_rows
from app.localization.assign_column import assign_columns
from app.localization.assign_subrow import assign_subrows
from app.utils.clip_utils import extract_product_embeddings
from app.utils.gpt_utils import call_gpt_with_images
from app.utils.io_utils import load_image_as_numpy
from app.core.model_loader import get_models
from app.planogram.compare_planogram import compare_planogram
from app.config import *

import os, time

def audit_image_pipline(
        planogram_path,  
        n_shelf, 
        id_store, 
        img_path, 
        data
    ):
    img = load_image_as_numpy(img_path)

    shelf_detector, product_detector = get_models()

    shelves = shelf_detector.detect(id_store, img)
    products = product_detector.detect(n_shelf, img)

    products = assign_rows(products, shelves)
    products = assign_columns(products)
    products = assign_subrows(products)

    for i, p in enumerate(products, start=1):
        p["id"] = i
    
    extract_product_embeddings(products, img)

    # product List[Dict], for p in products: {"store" : x, "shelf" : x, "id" : x, "bbox" : [x,x,x,x], "conf" : x, "col" : x, "row" : x, "subrow" : x, "embedding" : [x,x,x] }
    
    compare_planogram(products, data)

    return