MODEL = "google/gemma-3-27b-it"
RESULTS_N = 7 # 5

# OpenRouter prices (USD per 1M tokens)
PRICE_INPUT_PER_1M = 0.04
PRICE_OUTPUT_PER_1M = 0.15

# Columns to show in the context table sent to the LLM
COLS_SHOW = [
    "aine_kood",
    "nimi_et",
    "nimi_en",
    "eap",
    "semester",
    "oppeaste",
    "keel",
    "linn",
    "hindamisviis",
]

# Paths
DATA_DIR = "andmed"
DATA_CSV = f"{DATA_DIR}/puhtad_andmed.csv"
DATA_EMB = f"{DATA_DIR}/puhtad_andmed_embeddings.pkl"
HISTORY_DIR = "history"
