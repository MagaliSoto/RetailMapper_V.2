import os
import base64
import time
import logging
from typing import Optional, Union, List

from openai import OpenAI

from app.config import (
    GPT_MODEL,
    GPT_TEMPERATURE,
    GPT_MAX_TOKENS,
    GPT_SYSTEM_PROMPT
)

logger = logging.getLogger(__name__)
client = OpenAI()
counter = 0

def call_gpt_with_image(
    image_paths: Union[str, List[str]],
    user_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Call GPT with one or multiple images and optional text prompt.

    Parameters
    ----------
    image_paths : Union[str, List[str]]
        Path or list of paths to image files.
    user_prompt : Optional[str]
        User-level instruction sent along with the images.
    system_prompt : Optional[str]
        System-level instruction. Defaults to GPT_SYSTEM_PROMPT.
    temperature : Optional[float]
        Sampling temperature (ignored for certain models like gpt-5).
    max_tokens : Optional[int]
        Maximum output tokens override.

    Returns
    -------
    str
        Model textual response (stripped). Empty string if invalid structure.
    """

    start = time.perf_counter()

    if not image_paths:
        raise ValueError("image_paths must not be empty.")

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    if not isinstance(image_paths, list):
        raise TypeError("image_paths must be a string or list of strings.")

    logger.info(
        "Calling GPT with images",
        extra={"num_images": len(image_paths)}
    )

    content: List[dict] = []

    # Add user text instruction if provided
    if user_prompt:
        content.append({
            "type": "input_text",
            "text": user_prompt
        })

    # Encode images as base64 and append to content
    for image_path in image_paths:
        if not isinstance(image_path, str):
            raise TypeError("Each image path must be a string.")

        if not os.path.exists(image_path):
            logger.error(
                "Image not found",
                extra={"image_path": image_path}
            )
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
        "max_output_tokens": (
            max_tokens if max_tokens is not None else GPT_MAX_TOKENS
        ),
    }

    # Temperature is ignored for some models (e.g., gpt-5)
    if GPT_MODEL != "gpt-5":
        request["temperature"] = (
            temperature if temperature is not None else GPT_TEMPERATURE
        )

    try:
        response = client.responses.create(**request)
    except Exception:
        logger.exception("GPT API call failed")
        raise

    duration = time.perf_counter() - start

    logger.info(
        "GPT raw response received",
        extra={"duration_seconds": round(duration, 3)}
    )

    if not hasattr(response, "output_text"):
        logger.error(
            "GPT response missing output_text",
            extra={"response_repr": repr(response)}
        )
        return ""

    logger.info(
        "GPT output received",
        extra={"output_preview": response.output_text[:200]}
    )

    return response.output_text.strip()


def call_gpt_with_images(
    image_paths: Union[str, List[str]],
    user_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    max_retries: int = 1,
) -> str:
    """
    Call GPT with retry logic and fallback strategy.

    Retries the main prompt up to `max_retries` times.
    If all attempts fail or produce invalid labels,
    a fallback descriptive prompt is used.

    Parameters
    ----------
    image_paths : Union[str, List[str]]
        Image path or list of image paths.
    user_prompt : Optional[str]
        Primary user instruction.
    system_prompt : Optional[str]
        System-level instruction.
    max_retries : int
        Number of retry attempts before fallback.

    Returns
    -------
    str
        Validated label or fallback value.
    """
    global counter

    if max_retries <= 0:
        raise ValueError("max_retries must be greater than 0.")

    # Primary attempts
    for attempt in range(max_retries):
        try:
            result = call_gpt_with_image(
                image_paths=image_paths,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
            )

            logger.info(
                "GPT primary attempt completed",
                extra={"attempt": attempt + 1, "result": result}
            )

            if is_valid_label(result):
                return result.strip()

            logger.warning(
                "Invalid label from GPT (primary)",
                extra={"attempt": attempt + 1}
            )

        except Exception:
            logger.exception(
                "GPT primary attempt failed",
                extra={"attempt": attempt + 1}
            )

    logger.warning("All primary GPT attempts failed. Trying fallback prompt.")

    # Fallback attempts (2 times)
    fallback_prompt = (
        "Describí únicamente las características visuales del producto "
        "sin inventar marca ni tipo específico."
    )

    for fallback_attempt in range(2):
        try:
            result = call_gpt_with_image(
                image_paths=image_paths,
                user_prompt=fallback_prompt,
                system_prompt=system_prompt,
            )

            logger.info(
                "GPT fallback attempt completed",
                extra={"fallback_attempt": fallback_attempt + 1, "result": result}
            )

            if is_valid_label(result):
                return result.strip()

            logger.warning(
                "Invalid label from GPT (fallback)",
                extra={"fallback_attempt": fallback_attempt + 1}
            )

        except Exception:
            logger.exception(
                "GPT fallback attempt failed",
                extra={"fallback_attempt": fallback_attempt + 1}
            )

    # Hard fallback
    logger.error("Returning hard fallback label")

    counter += 1
    return f"producto genérico {counter}"



def is_valid_label(text: str) -> bool:
    """
    Validate label output returned by GPT.

    A valid label:
        - Is not empty
        - Is not "NO LABEL"
        - Contains at least one non-whitespace character

    Parameters
    ----------
    text : str
        GPT output text.

    Returns
    -------
    bool
        True if valid, False otherwise.
    """

    if not text:
        return False

    cleaned = text.strip()

    if cleaned.upper() == "NO LABEL":
        return False

    return len(cleaned) > 0

def call_gpt_with_text(
    user_prompt: str,
    max_tokens: int = 200,
) -> str:

    if not user_prompt:
        raise ValueError("user_prompt must not be empty.")

    response = client.responses.create(
        model=GPT_MODEL,  # gpt-5
        input=user_prompt,
        max_output_tokens=max_tokens,
        reasoning={"effort": "minimal"}  # 👈 IMPORTANTE
    )

    output_text = response.output_text or ""

    return output_text.strip()
