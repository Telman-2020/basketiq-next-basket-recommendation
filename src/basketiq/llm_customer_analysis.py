import joblib
import ollama
import pandas as pd

from src.basketiq.config import PROCESSED_DATA_DIR, TRAINING_DATA_FILE


MODEL_FILE = (
    PROCESSED_DATA_DIR
    / "tuned_models"
    / "weighted_soft_voting_boosting_models.joblib"
)

LOCAL_LLM_MODEL = "llama3.2:3b"

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


def load_data_and_model():
    """
    Load the engineered modelling dataset and the trained champion model.
    """
    if not TRAINING_DATA_FILE.exists():
        raise FileNotFoundError(f"Training data file not found: {TRAINING_DATA_FILE}")

    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Champion model file not found: {MODEL_FILE}")

    df = pd.read_parquet(TRAINING_DATA_FILE)
    model = joblib.load(MODEL_FILE)

    return df, model


def get_customer_recommendation_context(user_id: int, top_k: int = 10) -> dict:
    """
    Build a compact customer-level context for local LLM interpretation.
    """
    df, model = load_data_and_model()

    user_df = df[df["user_id"] == user_id].copy()

    if user_df.empty:
        raise ValueError(f"No historical product data found for user_id={user_id}")

    missing_features = [col for col in FEATURE_COLUMNS if col not in user_df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    user_df["recommendation_score"] = model.predict_proba(
        user_df[FEATURE_COLUMNS]
    )[:, 1]

    top_recommendations = (
        user_df.sort_values("recommendation_score", ascending=False)
        .head(top_k)
        [
            [
                "product_id",
                "product_name",
                "recommendation_score",
                "reordered_next_order",
                "user_product_total_orders",
                "user_product_reorder_rate",
                "orders_since_last_purchase",
                "product_reorder_rate",
                "user_product_purchase_share",
            ]
        ]
        .copy()
    )

    top_recommendations["recommendation_score"] = top_recommendations[
        "recommendation_score"
    ].round(4)

    top_recommendations["user_product_reorder_rate"] = top_recommendations[
        "user_product_reorder_rate"
    ].round(4)

    top_recommendations["product_reorder_rate"] = top_recommendations[
        "product_reorder_rate"
    ].round(4)

    top_recommendations["user_product_purchase_share"] = top_recommendations[
        "user_product_purchase_share"
    ].round(4)

    historical_summary = {
        "user_id": int(user_id),
        "candidate_products_seen": int(len(user_df)),
        "user_total_orders": float(user_df["user_total_orders"].max()),
        "user_avg_basket_size": round(float(user_df["user_avg_basket_size"].mean()), 2),
        "user_reorder_rate": round(float(user_df["user_reorder_rate"].mean()), 4),
        "avg_days_since_prior_order": round(
            float(user_df["user_avg_days_since_prior_order"].mean()), 2
        ),
        "actually_reordered_candidates": int(user_df["reordered_next_order"].sum()),
    }

    frequent_products = (
        user_df.sort_values("user_product_total_orders", ascending=False)
        .head(10)
        [
            [
                "product_name",
                "user_product_total_orders",
                "user_product_reorder_rate",
                "orders_since_last_purchase",
            ]
        ]
        .copy()
    )

    frequent_products["user_product_reorder_rate"] = frequent_products[
        "user_product_reorder_rate"
    ].round(4)

    recent_products = (
        user_df.sort_values("orders_since_last_purchase", ascending=True)
        .head(10)
        [
            [
                "product_name",
                "user_product_total_orders",
                "user_product_reorder_rate",
                "orders_since_last_purchase",
            ]
        ]
        .copy()
    )

    recent_products["user_product_reorder_rate"] = recent_products[
        "user_product_reorder_rate"
    ].round(4)

    return {
        "historical_summary": historical_summary,
        "top_recommendations": top_recommendations,
        "frequent_products": frequent_products,
        "recent_products": recent_products,
    }


def build_llm_prompt(context: dict) -> str:
    """
    Create a grounded prompt using only model outputs and customer-history tables.
    """
    historical_summary = context["historical_summary"]
    top_recommendations = context["top_recommendations"]
    frequent_products = context["frequent_products"]
    recent_products = context["recent_products"]

    prompt = f"""
You are a data scientist analysing a grocery next-basket recommendation model.

Use only the evidence provided in the tables below.
Do not invent customer details.
Do not claim the customer bought something unless it appears in the tables.
Keep the explanation concise, practical and business-focused.

Customer summary:
{historical_summary}

Top model recommendations:
{top_recommendations.to_markdown(index=False)}

Most frequently purchased historical products:
{frequent_products.to_markdown(index=False)}

Most recently purchased products:
{recent_products.to_markdown(index=False)}

Write the analysis using these sections:

1. Customer behaviour summary
2. Why the model recommended these products
3. Evidence from frequency, recency and reorder behaviour
4. Reliability of the recommendations
5. Possible business action
"""
    return prompt


def analyse_customer_with_local_llm(
    user_id: int,
    top_k: int = 10,
    model_name: str = LOCAL_LLM_MODEL,
) -> str:
    """
    Generate a local LLM explanation for a customer's recommendations.
    """
    context = get_customer_recommendation_context(
        user_id=user_id,
        top_k=top_k,
    )

    prompt = build_llm_prompt(context)

    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful retail data scientist. "
                    "You explain model outputs using only evidence from the supplied tables. "
                    "You avoid hallucination and clearly separate evidence from interpretation."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response["message"]["content"]


if __name__ == "__main__":
    df = pd.read_parquet(TRAINING_DATA_FILE)
    example_user_id = int(df["user_id"].iloc[0])

    print(f"Running local LLM customer analysis for user_id={example_user_id}")
    analysis = analyse_customer_with_local_llm(
        user_id=example_user_id,
        top_k=10,
    )

    print("\nLLM Customer Analysis")
    print("=" * 80)
    print(analysis)
