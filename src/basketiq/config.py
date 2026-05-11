from pathlib import Path


# Project root:
# basketiq-next-basket-recommendation/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Raw dataset files
ORDERS_FILE = RAW_DATA_DIR / "orders.csv"
ORDER_PRODUCTS_PRIOR_FILE = RAW_DATA_DIR / "order_products__prior.csv"
ORDER_PRODUCTS_TRAIN_FILE = RAW_DATA_DIR / "order_products__train.csv"
PRODUCTS_FILE = RAW_DATA_DIR / "products.csv"
AISLES_FILE = RAW_DATA_DIR / "aisles.csv"
DEPARTMENTS_FILE = RAW_DATA_DIR / "departments.csv"

# Processed output files
TRAINING_DATA_FILE = PROCESSED_DATA_DIR / "training_data.parquet"
MODEL_RESULTS_FILE = PROCESSED_DATA_DIR / "model_results.csv"