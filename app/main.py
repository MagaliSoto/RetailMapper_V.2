from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import shutil, uuid, os

from app.services.planogram_pipeline import process_image_pipeline
from app.core.model_loader import load_models
from app.utils.json_utils import save_products_to_json, save_groups_to_json
from app.config import DOWNLOAD_JSON, OUTPUT_FOLDER

app = FastAPI(title="Retail Mapper API")

@app.on_event("startup")
def startup_event():
    load_models()

@app.post("/process")
async def process(
    image: UploadFile = File(...),
    n_shelf: int = Form(...),
    id_store: int = Form(...)
):
    try:
        temp_dir = "tmp"
        os.makedirs(temp_dir, exist_ok=True)

        img_path = f"{temp_dir}/{uuid.uuid4()}.jpg"
        with open(img_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        products, data = process_image_pipeline(img_path, n_shelf, id_store)

        if DOWNLOAD_JSON:
            save_products_to_json(
                products=products,
                filename="products.json",
                output_folder=OUTPUT_FOLDER
            )
            save_groups_to_json(
                data_groups=data,
                filename="data.json",
                output_folder=OUTPUT_FOLDER
            )

        return {
            "status": "ok",
            "count": len(products),
            "products": products
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )
