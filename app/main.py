import logging
import os
import shutil
import uuid
import json

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from app.services.process_planogram_pipeline import process_image_pipeline
from app.services.audit_pipline import audit_image_pipeline
from app.core.model_loader import load_models
from app.utils.json_utils import save_products_to_json, save_groups_to_json
from app.config import DOWNLOAD_JSON, OUTPUT_FOLDER


def configure_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    log_format = (
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(funcName)s | "
        "%(message)s"
    )

    logger = logging.getLogger()
    logger.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))

    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/app.log")
    file_handler.setFormatter(logging.Formatter(log_format))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

configure_logging()

app = FastAPI(title="Retail Mapper API")


@app.on_event("startup")
def startup_event():
    logging.getLogger(__name__).info("Loading models at startup")
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

        products, data_groups = process_image_pipeline(
            img_path,
            n_shelf,
            id_store
        )

        if DOWNLOAD_JSON:
            save_products_to_json(
                products=products,
                filename="products.json",
                output_folder=OUTPUT_FOLDER
            )
            save_groups_to_json(
                data_groups=data_groups,
                filename="data_groups.json",
                output_folder=OUTPUT_FOLDER
            )

        return {
            "status": "ok",
            "count": len(products),
            "products": products
        }

    except Exception as e:
        logging.getLogger(__name__).exception("Error in /process endpoint")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )


@app.post("/audit")
async def audit(
    image: UploadFile = File(...),
    n_shelf: int = Form(...),
    id_store: int = Form(...),
    planogram_path: str = Form(...),
    data: str = Form(...)
):
    try:
        temp_dir = "tmp"
        os.makedirs(temp_dir, exist_ok=True)

        img_path = f"{temp_dir}/{uuid.uuid4()}.jpg"
        with open(img_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        parsed_data = json.loads(data)

        with open("output/data_groups.json", "r", encoding="utf-8") as f:
            label_ids_dict = json.load(f)

        result_data = audit_image_pipeline(
            n_shelf=n_shelf,
            id_store=id_store,
            img_path=img_path,
            label_ids_dict=label_ids_dict,
            planogram_data=parsed_data
        )

        if DOWNLOAD_JSON:
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            save_groups_to_json(
                data_groups=result_data,
                filename="audit_data.json",
                output_folder=OUTPUT_FOLDER
            )

        return {
            "status": "ok",
            "products": result_data
        }

    except Exception as e:
        logging.getLogger(__name__).exception("Error in /audit endpoint")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "detail": str(e)
            }
        )
