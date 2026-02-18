"""
Integration test for the /process API endpoint.

This test:
- Sends an image to the API
- Triggers the detection + grouping pipeline
- Prints the resulting JSON response

The FastAPI server must be running locally.
"""

import requests

API_URL = "http://localhost:8000/process"

IMAGE_PATH = "input_images_test/imgGondola2.jpeg"
N_SHELF = 1
ID_STORE = 2


def main():
    """
    Sends image and metadata to the /process endpoint.
    """

    with open(IMAGE_PATH, "rb") as img:
        files = {
            "image": ("imgGondola2.jpeg", img, "image/jpeg")
        }

        data = {
            "n_shelf": N_SHELF,
            "id_store": ID_STORE
        }

        print("Sending request to API...")
        response = requests.post(API_URL, files=files, data=data)

    print("Status code:", response.status_code)

    try:
        json_resp = response.json()
        print("Response JSON:")
        print(json_resp)
    except Exception:
        print("Raw response:")
        print(response.text)


if __name__ == "__main__":
    main()
