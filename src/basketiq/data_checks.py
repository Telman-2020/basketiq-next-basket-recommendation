import pandas as pd

from src.basketiq.config import (
    ORDERS_FILE,
    ORDER_PRODUCTS_PRIOR_FILE,
    ORDER_PRODUCTS_TRAIN_FILE,
    PRODUCTS_FILE,
    AISLES_FILE,
    DEPARTMENTS_FILE,
)


REQUIRED_FILES = [
    ORDERS_FILE,
    ORDER_PRODUCTS_PRIOR_FILE,
    ORDER_PRODUCTS_TRAIN_FILE,
    PRODUCTS_FILE,
    AISLES_FILE,
    DEPARTMENTS_FILE,
]


def check_required_files() -> None:
    missing_files = [file for file in REQUIRED_FILES if not file.exists()]

    if missing_files:
        print("Missing required files:")
        for file in missing_files:
            print(f"- {file}")
        raise FileNotFoundError("Some required raw data files are missing.")

    print("All required raw data files exist.")


def preview_raw_data() -> None:
    files = {
        "orders": ORDERS_FILE,
        "order_products_prior": ORDER_PRODUCTS_PRIOR_FILE,
        "order_products_train": ORDER_PRODUCTS_TRAIN_FILE,
        "products": PRODUCTS_FILE,
        "aisles": AISLES_FILE,
        "departments": DEPARTMENTS_FILE,
    }

    for name, path in files.items():
        df = pd.read_csv(path, nrows=5)
        print(f"\n{name}")
        print("-" * 50)
        print(df.head())


if __name__ == "__main__":
    check_required_files()
    preview_raw_data()