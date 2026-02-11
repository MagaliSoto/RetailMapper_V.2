# RetailMapper v.2

**Product detection, shelf localization and semantic labeling as a microservice**

RetailMapper v.2 is a Dockerized REST API that processes retail shelf images and returns a structured JSON describing each detected product.  
For every product, the system computes spatial location (row, column, subrow), visual embeddings, and a semantic label generated via OpenAI.

This version is **API-only** and designed to be consumed as a microservice in larger retail analytics pipelines.

---

## Table of Contents

- [About](#about)
- [Architecture Overview](#architecture-overview)
- [API Usage](#api-usage)
- [Docker Setup](#docker-setup)
- [Configuration](#configuration)
- [Environment Variables](#environment-variables)
- [Response Format](#response-format)
- [Project Structure](#project-structure)
- [Tests](#tests)
- [Upcoming Features](#upcoming-features)
- [Contact](#contact)

---

## About

RetailMapper v.2 receives a single shelf image and metadata, then executes a full computer vision pipeline:

1. Detects shelves and products using YOLO-based models  
2. Assigns a unique ID to each detected product  
3. Computes spatial location:
   - `row`
   - `col`
   - `subrow`
4. Extracts **CLIP embeddings** for each product crop  
5. Generates a **semantic label in Spanish** using OpenAI (image-based prompting)  
6. Aggregates all information into a structured JSON response

The service is stateless from the client perspective and communicates exclusively via HTTP.

---

## Architecture Overview

The processing pipeline follows this flow:

```
Input image
   ↓
Shelf detection
   ↓
Product detection
   ↓
ID assignment
   ↓
Row / Column / Subrow localization
   ↓
CLIP embedding extraction
   ↓
OpenAI-based product labeling
   ↓
JSON response
```

All steps are executed synchronously inside a single API call.

---

## API Usage

### Endpoint

**POST `/process`**

This is the only exposed endpoint.

### Request format

The endpoint expects a `multipart/form-data` request with the following fields:

| Field | Type | Required | Description |
|------|------|----------|-------------|
| `image` | File | ✅ | Shelf image to process |
| `n_shelf` | int | ✅ | Shelf number |
| `id_store` | int | ✅ | Store identifier |

---

## Docker Setup

### Build the image

```bash
docker build -t retail-mapper-api .
```

### Run the container

```bash
docker run -p 8000:8000 --env-file .env retail-mapper-api
```
### Run the container and download JSON

```bash
mkdir output
docker run -p 8000:8000 -v ${PWD}/output:/app/output --env-file .env retail-mapper-api
```

---

## Configuration

Main configuration values are defined in `app/config.py`.

---

## Environment Variables

The service requires access to OpenAI for product labeling.

```env
OPENAI_API_KEY=your_api_key_here
```

---

## Response Format

```json
{
  "status": "ok",
  "count": 1,
  "products": [
    {
      "bbox": [57, 209, 181, 417],
      "conf": 0.89,
      "embeddings": [ ... ],
      "label": "botella de shampoo",
      "shelf": 1,
      "row": 3,
      "col": 1,
      "subrow": 1,
      "id": 1
    }
  ]
}
```

---

## Project Structure

```
.
├── Dockerfile
├── requirements.txt
├── .env
├── app/
│   ├── main.py
│   ├── config.py
│   ├── core/
│   ├── detectors/
│   ├── localization/
│   ├── services/
│   ├── utils/
│   └── tests/
├── models/
└── input_images_test/
```

---

## Tests

Tests are internal and intended for development validation.

To run them:
1. Move the `tests/` folder to the project root  
2. Execute:

```bash
python tests/test_api.py
```

---

## Upcoming Features

A second microservice is planned under:

```
services/audit_planogram.py
```

This service will be responsible for **auditing a reference planogram against real shelf images**, enabling:

- Product presence / absence validation
- Layout compliance checks
- Automated planogram auditing workflows

Once implemented, this service will be documented and integrated into the README.

---

## Contact

**Magali Soto**  
Software Developer  
📧 sotomagali265@gmail.com
