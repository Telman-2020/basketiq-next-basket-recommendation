import joblib
import pandas as pd

from src.basketiq.config import PROCESSED_DATA_DIR, TRAINING_DATA_FILE


MODEL_FILE = PROCESSED_DATA_DIR / "tuned_models" / "weighted_soft_voting_boosting_models.joblib"

FEATURE_COLUMNS = [
    "user_product_total_orders",
    "user_product_total_reorders",
    "user_product_avg_cart_order",
    "user_product_last_order_number",
    "user_total_orders",
    "user_avg_days_since_prior_order",
    "user_avg_basket_size",
    "user_max_basket_size",
    "user_total_products",
    "user_total_reorders",
    "user_reorder_rate",
    "product_total_purchases",
    "product_total_reorders",
    "product_avg_add_to_cart_order",
    "product_reorder_rate",
    "product_first_cart_count",
    "product_first_cart_rate",
    "user_product_reorder_rate",
    "orders_since_last_purchase",
    "user_product_purchase_share",
    "log1p_user_product_total_orders",
    "log1p_user_product_total_reorders",
    "log1p_user_product_avg_cart_order",
    "log1p_user_product_last_order_number",
    "log1p_user_total_orders",
    "log1p_user_avg_basket_size",
    "log1p_user_max_basket_size",
    "log1p_user_total_products",
    "log1p_user_total_reorders",
    "log1p_product_total_purchases",
    "log1p_product_total_reorders",
    "log1p_product_avg_add_to_cart_order",
    "log1p_product_first_cart_count",
    "log1p_orders_since_last_purchase",
]


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")

    return joblib.load(MODEL_FILE)


def recommend_for_user(user_id: int, top_k: int = 10) -> pd.DataFrame:
    df = pd.read_parquet(TRAINING_DATA_FILE)

    user_df = df[df["user_id"] == user_id].copy()

    if user_df.empty:
        raise ValueError(f"No historical product data found for user_id={user_id}")

    model = load_model()

    scores = model.predict_proba(user_df[FEATURE_COLUMNS])[:, 1]

    user_df["recommendation_score"] = scores

    output_columns = [
        "user_id",
        "product_id",
        "product_name",
        "aisle_id",
        "department_id",
        "recommendation_score",
        "user_product_total_orders",
        "user_product_reorder_rate",
        "orders_since_last_purchase",
    ]

    recommendations = (
        user_df[output_columns]
        .sort_values("recommendation_score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    return recommendations


if __name__ == "__main__":
    data = pd.read_parquet(TRAINING_DATA_FILE)

    example_user_id = int(data["user_id"].iloc[0])

    print(f"Generating top 10 recommendations for user_id={example_user_id}")

    recommendations = recommend_for_user(
        user_id=example_user_id,
        top_k=10,
    )

    print(recommendations)