# RetailMapper v.2

**Product detection, shelf localization, planogram auditing and task
automation as a microservice**

RetailMapper v.2 is a Dockerized REST API that processes retail shelf
images and performs full planogram compliance auditing.\
For every detected product, the system computes spatial location (row,
column, subrow), visual embeddings, and a semantic label generated via
OpenAI.

In addition, the system can audit real shelf images against a reference
planogram and generate structured compliance reports, tasks, and scoring
metrics.

This version is **API-only** and designed to be consumed as a
microservice in larger retail analytics pipelines.

------------------------------------------------------------------------

## Table of Contents

-   [About](#about)
-   [Architecture Overview](#architecture-overview)
-   [Audit Pipeline](#audit-pipeline)
-   [Task Engine](#task-engine)
-   [API Usage](#api-usage)
-   [Docker Setup](#docker-setup)
-   [Configuration](#configuration)
-   [Environment Variables](#environment-variables)
-   [Response Format](#response-format)
-   [JSON Outputs](#json-outputs)
-   [Project Structure](#project-structure)
-   [Performance Considerations](#performance-considerations)
-   [Tests](#tests)
-   [Contact](#contact)

------------------------------------------------------------------------

## About

RetailMapper v.2 receives a shelf image and metadata, then executes a
full computer vision pipeline:

1.  Detects shelves and products using YOLO-based models\
2.  Assigns a unique ID to each detected product\
3.  Computes spatial location:
    -   `row`
    -   `col`
    -   `subrow`
4.  Extracts **CLIP embeddings** for each product crop\
5.  Generates a **semantic label in Spanish** using OpenAI (image-based
    prompting)\
6.  Aggregates all information into a structured JSON response

The generated output can be used as a **digital reference planogram**.

The system can then audit new shelf images against this reference,
detecting:

-   Missing products\
-   Unexpected products\
-   Misplaced products\
-   Correctly placed products

The service is stateless from the client perspective and communicates
exclusively via HTTP.

------------------------------------------------------------------------

## Architecture Overview

### Planogram Processing Pipeline

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
    products.json + data_groups.json

All steps are executed synchronously inside a single API call.

------------------------------------------------------------------------

## Audit Pipeline

The audit pipeline compares a new shelf image against a previously
generated planogram.

    New shelf image
       ↓
    Shelf + product detection
       ↓
    Spatial localization
       ↓
    Matching against reference planogram
       ↓
    Classification:
          - match
          - missing
          - unexpected
          - different_location
       ↓
    compare_planogram.json
       ↓
    Task Engine
       ↓
    tasks.json (final output)

The compliance score is calculated as:

    score = match / total_expected

The summary paragraph is generated using GPT.

------------------------------------------------------------------------

## Task Engine

Located in:

    app/tasks/
       ├── task_builder.py
       ├── task_manager.py

It generates:

-   A structured list of corrective tasks\
-   A compliance score\
-   A GPT-generated executive summary

This allows automatic transformation of audit results into actionable
store-level instructions.

------------------------------------------------------------------------

## API Usage

### Endpoints

**POST `/process`**\
Generates the reference planogram.

**POST `/audit`**\
Audits a new image against the reference planogram.

------------------------------------------------------------------------

### Request format

Both endpoints expect `multipart/form-data`:

  Field        Type   Required   Description
  ------------ ------ ---------- ------------------
  `image`      File   ✅         Shelf image
  `n_shelf`    int    ✅         Shelf number
  `id_store`   int    ✅         Store identifier

Reference JSON files are loaded internally during audit.

------------------------------------------------------------------------

## Docker Setup

### Build the image

``` bash
docker build -t retail-mapper-api .
```

### Run the container

``` bash
docker run -p 8000:8000 --env-file .env retail-mapper-api
```

### Run with output folder

``` bash
mkdir output
docker run -p 8000:8000 -v ${PWD}/output:/app/output --env-file .env retail-mapper-api
```

### Run with logs

``` bash
docker run -p 8000:8000 -v ${PWD}/output:/app/output -v ${PWD}/logs:/app/logs --env-file .env retail-mapper-api
```

------------------------------------------------------------------------

## Configuration

Main configuration values are defined in `app/config.py`.

------------------------------------------------------------------------

## Environment Variables

The service requires access to OpenAI for:

-   Product labeling\
-   Audit summary generation

``` env
OPENAI_API_KEY=your_api_key_here
```

------------------------------------------------------------------------

## Response Format

### `/process`

``` json
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

------------------------------------------------------------------------

### `/audit`

``` json
{
  "status": "ok",
  "tasks": [
    "Shelf 1 - Restock product X",
    "Shelf 1 - Move product Y to row 2 column 3"
  ],
  "score": 0.85,
  "summary": "The shelf presents high compliance but requires corrective actions."
}
```

------------------------------------------------------------------------

## JSON Outputs

During execution the system generates:

-   `products.json`\
-   `data_groups.json`\
-   `compare_planogram.json`\
-   `tasks.json`

------------------------------------------------------------------------

## Project Structure

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
    │   │    ├── process_planogram_pipeline.py
    │   │    ├── audit_pipeline.py
    │   ├── tasks/
    │   │    ├── task_builder.py
    │   │    ├── task_manager.py
    │   ├── utils/
    │   └── tests/
    ├── models/
    ├── output/
    └── input_images_test/

------------------------------------------------------------------------

## Performance Considerations

The `/audit` endpoint is synchronous and takes approximately **2 minutes
per request** due to detection, matching, and GPT reasoning.

------------------------------------------------------------------------

## Tests

Tests are internal and intended for development validation.

To run them:

1.  Move the `tests/` folder to the project root\
2.  Execute:

``` bash
python tests/test_api.py
```

------------------------------------------------------------------------

## Contact

**Magali Soto**\
Software Developer\
📧 sotomagali265@gmail.com
