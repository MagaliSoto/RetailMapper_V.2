# =====================
# Paths & folders
# =====================

INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output"
OUTPUT_FOLDER_PRODUCT = "output_products"

SHELF_MODEL_PATH = "models/shelf-model.pt"
PRODUCT_MODEL_PATH = "models/product-model.pt"

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
# =====================
# Detection parameters
# =====================

CONF_THRESHOLD = 0.4
DEBUG = True  # Enable verbose debug output
DOWNLOAD_JSON = True

# =====================
# Visualization
# =====================

COLOR_SHELF = (255, 0, 0)     # Red
COLOR_PRODUCT = (0, 255, 0)   # Green


# =====================
# GPT configuration
# =====================

GPT_MODEL = "gpt-5"
GPT_TEMPERATURE = 0.2
GPT_MAX_TOKENS = 800

GPT_SYSTEM_PROMPT = """
Analizá cuidadosamente el producto que aparece en la imagen.

Tu tarea es devolver UN SOLO label corto en español.

Reglas estrictas:
- Si tenés MÁS del 90% de confianza en la marca y el tipo exacto de producto, devolvé:
  "marca + tipo de producto"
  Ejemplo: "Dove shampoo anticaspa"

- Si NO alcanzás ese nivel de confianza pero podés identificar claramente el tipo de producto,
  devolvé una descripción genérica SIN marca.
  Ejemplos:
  - "botella de shampoo"
  - "barra de chocolate"
  - "paquete de galletitas"
  - "bebida gaseosa en lata"

- Si NO alcanzás suficiente confianza para identificar marca o tipo exacto,
  describí el producto visualmente de forma concreta.
  Ejemplos:
  - "envase plástico con tapa marrón"
  - "botella transparente con líquido amarillo"
  - "paquete rojo con letras blancas"

- Solo si la imagen es completamente ilegible o no muestra un producto,
  devolvé exactamente:
  NO LABEL


- No inventes marcas ni sabores.
- No hagas suposiciones.
- No incluyas explicaciones, comentarios ni puntuación extra.
- Devolvé solo el label final, en una sola línea.
"""

