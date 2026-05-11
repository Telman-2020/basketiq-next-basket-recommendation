import pandas as pd

from src.basketiq.config import (
    ORDERS_FILE,
    ORDER_PRODUCTS_PRIOR_FILE,
    ORDER_PRODUCTS_TRAIN_FILE,
    PRODUCTS_FILE,
    PROCESSED_DATA_DIR,
    TRAINING_DATA_FILE,
)


def load_raw_data():
    """
    Load the raw Instacart datasets required for feature engineering.
    """
    orders = pd.read_csv(ORDERS_FILE)
    prior = pd.read_csv(ORDER_PRODUCTS_PRIOR_FILE)
    train = pd.read_csv(ORDER_PRODUCTS_TRAIN_FILE)
    products = pd.read_csv(PRODUCTS_FILE)

    return orders, prior, train, products


def build_training_data(sample_users: int = 5000):
    """
    Build the supervised user-product training table.

    Each row represents one customer-product pair.

    The model learns to answer:

        Will this customer reorder this product in the next labelled order?

    Target:
        reordered_next_order = 1 if the customer reordered the product
        in the next labelled order.
        reordered_next_order = 0 otherwise.

    Notes on missing values:
        Some missing values are created intentionally by left joins.
        For example, if a historical user-product pair does not appear
        in the user's labelled next order, the target is missing after
        the join. In this business context, that missing value means
        "not reordered", so it is filled with 0.

        For engineered numeric features, 0 is also meaningful when it
        represents no observed count, no reorder event, or no matching
        purchase history.
    """

    print("Loading raw data...")
    orders, prior, train, products = load_raw_data()

    print("Selecting users with labelled train orders...")
    train_orders = orders.loc[
        orders["eval_set"] == "train",
        ["user_id", "order_id"],
    ]

    if sample_users is not None:
        sample_size = min(sample_users, train_orders["user_id"].nunique())

        sample_user_ids = (
            train_orders["user_id"]
            .drop_duplicates()
            .sample(n=sample_size, random_state=42)
        )

        train_orders = train_orders[train_orders["user_id"].isin(sample_user_ids)]
    else:
        sample_user_ids = train_orders["user_id"].drop_duplicates()

    print(f"Users selected: {train_orders['user_id'].nunique():,}")

    print("Preparing historical prior orders...")
    prior_orders = orders.loc[
        (orders["eval_set"] == "prior")
        & (orders["user_id"].isin(sample_user_ids)),
        ["user_id", "order_id", "order_number", "days_since_prior_order"],
    ]

    print("Joining prior products with prior orders...")
    prior_full = prior.merge(
        prior_orders,
        on="order_id",
        how="inner",
    )

    print("Creating user-product historical features...")
    user_product_features = (
        prior_full.groupby(["user_id", "product_id"])
        .agg(
            user_product_total_orders=("order_id", "nunique"),
            user_product_total_reorders=("reordered", "sum"),
            user_product_avg_cart_order=("add_to_cart_order", "mean"),
            user_product_last_order_number=("order_number", "max"),
        )
        .reset_index()
    )

    print("Creating user-level features...")
    print("Creating user-level features...")
    user_order_sizes = (
        prior_full.groupby(["user_id", "order_id"])
        .size()
        .reset_index(name="basket_size")
    )

    user_basket_features = (
        user_order_sizes.groupby("user_id")
        .agg(
            user_avg_basket_size=("basket_size", "mean"),
            user_max_basket_size=("basket_size", "max"),
        )
        .reset_index()
    )

    user_features = (
        prior_orders.groupby("user_id")
        .agg(
            user_total_orders=("order_id", "nunique"),
            user_avg_days_since_prior_order=("days_since_prior_order", "mean"),
        )
        .reset_index()
    )

    user_reorder_features = (
        prior_full.groupby("user_id")
        .agg(
            user_total_products=("product_id", "count"),
            user_total_reorders=("reordered", "sum"),
        )
        .reset_index()
    )

    user_reorder_features["user_reorder_rate"] = (
        user_reorder_features["user_total_reorders"]
        / user_reorder_features["user_total_products"]
    )

    user_features = user_features.merge(
        user_basket_features,
        on="user_id",
        how="left",
    )

    user_features = user_features.merge(
        user_reorder_features,
        on="user_id",
        how="left",
    )

    print("Creating product-level features...")
    product_features = (
        prior_full.groupby("product_id")
        .agg(
            product_total_purchases=("order_id", "count"),
            product_total_reorders=("reordered", "sum"),
            product_avg_add_to_cart_order=("add_to_cart_order", "mean"),
        )
        .reset_index()
        )

    product_features["product_reorder_rate"] = (
        product_features["product_total_reorders"]
        / product_features["product_total_purchases"]
    )

    first_cart_counts = (
        prior_full[prior_full["add_to_cart_order"] == 1]
        .groupby("product_id")
        .size()
        .reset_index(name="product_first_cart_count")
    )

    product_features = product_features.merge(
        first_cart_counts,
        on="product_id",
        how="left",
    )

    product_features["product_first_cart_count"] = (
        product_features["product_first_cart_count"].fillna(0)
    )

    product_features["product_first_cart_rate"] = (
        product_features["product_first_cart_count"]
        / product_features["product_total_purchases"]
    )
    print("Creating next-order target labels...")
    train_labels = train.merge(
        train_orders,
        on="order_id",
        how="inner",
    )

    train_labels = train_labels[["user_id", "product_id", "reordered"]].rename(
        columns={"reordered": "reordered_next_order"}
    )

    print("Joining all features into one modelling table...")
    training_data = user_product_features.merge(
        user_features,
        on="user_id",
        how="left",
    )

    training_data = training_data.merge(
        product_features,
        on="product_id",
        how="left",
    )

    training_data = training_data.merge(
        products,
        on="product_id",
        how="left",
    )

    training_data = training_data.merge(
        train_labels,
        on=["user_id", "product_id"],
        how="left",
    )

    print("Creating additional engineered features...")

    training_data["reordered_next_order"] = (
        training_data["reordered_next_order"]
        .fillna(0)
        .astype(int)
    )

    training_data["user_product_reorder_rate"] = (
        training_data["user_product_total_reorders"]
        / training_data["user_product_total_orders"]
    )

    training_data["orders_since_last_purchase"] = (
        training_data["user_total_orders"]
        - training_data["user_product_last_order_number"]
    )

    training_data["user_product_purchase_share"] = (
    training_data["user_product_total_orders"]
    / training_data["user_total_orders"]
    )

    print("Creating log-transformed features for skewed count variables...")

    log_transform_columns = [
        "user_product_total_orders",
        "user_product_total_reorders",
        "user_product_avg_cart_order",
        "user_product_last_order_number",
        "user_total_orders",
        "user_avg_basket_size",
        "user_max_basket_size",
        "user_total_products",
        "user_total_reorders",
        "product_total_purchases",
        "product_total_reorders",
        "product_avg_add_to_cart_order",
        "product_first_cart_count",
        "orders_since_last_purchase",
    ]

    for col in log_transform_columns:
        training_data[f"log1p_{col}"] = training_data[col].clip(lower=0).apply(
            lambda x: pd.NA if pd.isna(x) else x
        )
        training_data[f"log1p_{col}"] = (
            training_data[f"log1p_{col}"]
            .astype(float)
            .pipe(lambda s: s.mask(s < 0, 0))
        )
        training_data[f"log1p_{col}"] = training_data[f"log1p_{col}"].apply(
            lambda x: __import__("math").log1p(x)
        )

    numeric_columns = training_data.select_dtypes(include=["number"]).columns

    training_data[numeric_columns] = training_data[numeric_columns].fillna(0)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Saving training data to: {TRAINING_DATA_FILE}")
    training_data.to_parquet(TRAINING_DATA_FILE, index=False)

    print("Feature engineering complete.")
    print(f"Rows: {len(training_data):,}")
    print(f"Columns: {len(training_data.columns):,}")
    print(f"Target positive rate: {training_data['reordered_next_order'].mean():.4f}")

    return training_data


if __name__ == "__main__":
    build_training_data(sample_users= 5000)