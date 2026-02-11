import os
import base64
from typing import Optional, Union, List
from openai import OpenAI
import logging
import time
from app.config import (
    GPT_MODEL,
    GPT_TEMPERATURE,
    GPT_MAX_TOKENS,
    GPT_SYSTEM_PROMPT
)

logger = logging.getLogger(__name__)

client = OpenAI()


def call_gpt_with_image(
    image_paths: Union[str, List[str]],
    user_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:

    start = time.perf_counter()

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    logger.info(f"Calling GPT with {len(image_paths)} images")

    content = []

    if user_prompt:
        content.append({
            "type": "input_text",
            "text": user_prompt
        })

    for image_path in image_paths:
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{image_base64}"
        })

    request = {
        "model": GPT_MODEL,
        "input": [
            {
                "role": "system",
                "content": system_prompt if system_prompt else GPT_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": content
            }
        ],
        "max_output_tokens": max_tokens if max_tokens is not None else GPT_MAX_TOKENS,
    }

    if GPT_MODEL != "gpt-5":
        request["temperature"] = (
            temperature if temperature is not None else GPT_TEMPERATURE
        )

    try:
        response = client.responses.create(**request)
    except Exception as e:
        logger.exception("GPT API call failed")
        raise

    duration = time.perf_counter() - start
    logger.info(f"GPT raw response received in {duration:.3f}s")

    if not hasattr(response, "output_text"):
        logger.error(f"GPT response has no output_text: {response}")
        return ""

    logger.info(f"GPT output: {response.output_text}")

    return response.output_text.strip()

def call_gpt_with_images(
    image_paths: Union[str, List[str]],
    user_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    max_retries: int = 3,
) -> str:

    for attempt in range(max_retries):
        try:
            result = call_gpt_with_image(
                image_paths=image_paths,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
            )

            logger.info(f"GPT attempt {attempt+1} result: {result}")

            if is_valid_label(result):
                return result.strip()

            logger.warning("Invalid label from GPT")

        except Exception as e:
            logger.exception(f"GPT attempt {attempt+1} failed")

    logger.warning("All GPT attempts failed. Trying fallback prompt.")

    fallback_prompt = (
        "Describí únicamente las características visuales del producto "
        "sin inventar marca ni tipo específico."
    )

    try:
        result = call_gpt_with_image(
            image_paths=image_paths,
            user_prompt=fallback_prompt,
            system_prompt=system_prompt,
        )

        if is_valid_label(result):
            return result.strip()

    except Exception:
        logger.exception("Fallback GPT call failed")

    logger.error("Returning hard fallback: producto genérico")
    return "producto genérico"


def is_valid_label(text: str) -> bool:
    """
    Validates label output from GPT.
    """

    if not text:
        return False

    cleaned = text.strip()

    # "NO LABEL" is never valid
    if cleaned.upper() == "NO LABEL":
        return False

    return len(cleaned) > 0
