"""
Integration test for the /audit API endpoint.

This script sends a real HTTP POST request to the running FastAPI server
and validates the audit pipeline behavior end-to-end.

What this test does:
- Loads a real image file
- Loads a planogram JSON file
- Sends them to the /audit endpoint
- Prints the response status and JSON output

Requirements:
- FastAPI server must be running locally on port 8000
- Required JSON files must exist
"""

import requests
import json

# ============================================================
# CONFIGURATION VARIABLES (Modify these for testing)
# ============================================================

API_URL = "http://localhost:8000/audit"

IMAGE_PATH = "input_images_test/img4.jpeg"
PLANOGRAM_PATH = "input_images_test/img1.jpeg"
DATA_JSON_PATH = "output/data.json"

N_SHELF = 1
ID_STORE = 2


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """
    Sends a request to the /audit endpoint with image and planogram data.
    """

    # Load JSON data that simulates planogram input
    with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
        real_data = json.load(f)

    # Open image as binary for multipart/form-data request
    with open(IMAGE_PATH, "rb") as img:
        files = {
            "image": ("audit_image.jpg", img, "image/jpeg")
        }

        # Convert planogram object into JSON string
        data = {
            "n_shelf": N_SHELF,
            "id_store": ID_STORE,
            "planogram_path": PLANOGRAM_PATH,
            "data": json.dumps(real_data)
        }

        print("Sending audit request to API...")
        response = requests.post(API_URL, files=files, data=data)

    print("Status code:", response.status_code)

    # Attempt to parse JSON response
    try:
        json_resp = response.json()
        print("\nResponse JSON:")
        print(json.dumps(json_resp, indent=2, ensure_ascii=False))
    except Exception:
        print("\nRaw response:")
        print(response.text)


if __name__ == "__main__":
    main()
