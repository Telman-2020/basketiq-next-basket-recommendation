import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.basketiq.config import PROCESSED_DATA_DIR, TRAINING_DATA_FILE
from src.basketiq.llm_customer_analysis import analyse_customer_with_local_llm


MODEL_FILE = (
    PROCESSED_DATA_DIR
    / "tuned_models"
    / "weighted_soft_voting_boosting_models.joblib"
)

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


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_parquet(TRAINING_DATA_FILE)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


def generate_recommendations(
    df: pd.DataFrame,
    model,
    user_id: int,
    top_k: int,
) -> pd.DataFrame:
    user_df = df[df["user_id"] == user_id].copy()

    if user_df.empty:
        return pd.DataFrame()

    user_df["recommendation_score"] = model.predict_proba(
        user_df[FEATURE_COLUMNS]
    )[:, 1]

    output_columns = [
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

    recommendations = (
        user_df[output_columns]
        .sort_values("recommendation_score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    recommendations = recommendations.rename(
        columns={
            "reordered_next_order": "actual_reordered_next_order",
        }
    )

    numeric_columns = [
        "recommendation_score",
        "user_product_reorder_rate",
        "product_reorder_rate",
        "user_product_purchase_share",
    ]

    for col in numeric_columns:
        recommendations[col] = recommendations[col].round(4)

    return recommendations


def get_customer_summary(df: pd.DataFrame, user_id: int) -> dict:
    user_df = df[df["user_id"] == user_id].copy()

    return {
        "candidate_products": int(len(user_df)),
        "user_total_orders": int(user_df["user_total_orders"].max()),
        "avg_basket_size": round(float(user_df["user_avg_basket_size"].mean()), 2),
        "user_reorder_rate": round(float(user_df["user_reorder_rate"].mean()), 4),
        "actual_reordered_candidates": int(user_df["reordered_next_order"].sum()),
    }


def get_frequent_products(df: pd.DataFrame, user_id: int, n: int = 10) -> pd.DataFrame:
    user_df = df[df["user_id"] == user_id].copy()

    frequent = (
        user_df.sort_values("user_product_total_orders", ascending=False)
        .head(n)
        [
            [
                "product_name",
                "user_product_total_orders",
                "user_product_reorder_rate",
                "orders_since_last_purchase",
            ]
        ]
        .reset_index(drop=True)
    )

    frequent["user_product_reorder_rate"] = frequent[
        "user_product_reorder_rate"
    ].round(4)

    return frequent


def get_recent_products(df: pd.DataFrame, user_id: int, n: int = 10) -> pd.DataFrame:
    user_df = df[df["user_id"] == user_id].copy()

    recent = (
        user_df.sort_values("orders_since_last_purchase", ascending=True)
        .head(n)
        [
            [
                "product_name",
                "user_product_total_orders",
                "user_product_reorder_rate",
                "orders_since_last_purchase",
            ]
        ]
        .reset_index(drop=True)
    )

    recent["user_product_reorder_rate"] = recent[
        "user_product_reorder_rate"
    ].round(4)

    return recent


st.set_page_config(
    page_title="BasketIQ Recommendations",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 BasketIQ: Next-Basket Recommendation System")

st.write(
    """
    BasketIQ predicts which grocery products a customer is likely to reorder
    in their next basket. The app combines a trained supervised machine learning
    recommender with a local LLM explanation layer.
    """
)

with st.expander("How this demo works", expanded=False):
    st.write(
        """
        The machine learning model scores candidate products for the selected
        customer and ranks them by reorder probability. The optional local LLM
        analysis uses the customer history, recommendation output, frequency,
        recency and reorder behaviour to produce a business-friendly explanation.
        """
    )

df = load_data()
model = load_model()

user_ids = sorted(df["user_id"].unique())

st.sidebar.header("Recommendation Settings")

selected_user_id = st.sidebar.selectbox(
    "Select customer user_id",
    user_ids,
)

top_k = st.sidebar.radio(
    "Number of recommendations",
    [5, 10],
    index=1,
)

st.sidebar.header("Local LLM Settings")

enable_llm = st.sidebar.checkbox(
    "Enable local LLM analysis",
    value=False,
)

llm_model_name = st.sidebar.text_input(
    "Ollama model name",
    value="llama3.2:3b",
)

st.sidebar.caption(
    "Requires Ollama running locally and the selected model downloaded."
)

generate_button = st.sidebar.button("Generate Recommendations")

if generate_button:
    recommendations = generate_recommendations(
        df=df,
        model=model,
        user_id=selected_user_id,
        top_k=top_k,
    )

    if recommendations.empty:
        st.error("No product history found for this user.")
    else:
        summary = get_customer_summary(df, selected_user_id)

        st.subheader(f"Customer Summary: User {selected_user_id}")

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Candidate products", summary["candidate_products"])
        col2.metric("Total prior orders", summary["user_total_orders"])
        col3.metric("Avg basket size", summary["avg_basket_size"])
        col4.metric("User reorder rate", summary["user_reorder_rate"])
        col5.metric("Actual reordered candidates", summary["actual_reordered_candidates"])

        st.subheader(f"Top {top_k} Product Recommendations")

        st.dataframe(
            recommendations,
            use_container_width=True,
        )

        st.markdown("### Recommendation Interpretation")

        top_product = recommendations.iloc[0]

        st.write(
            f"""
            The highest-ranked product is **{top_product['product_name']}**
            with a recommendation score of **{top_product['recommendation_score']}**.

            The table includes the model score, the actual target where available
            for offline validation, frequency, recency and reorder-rate signals.
            """
        )

        left_col, right_col = st.columns(2)

        with left_col:
            st.markdown("### Most Frequent Historical Products")
            st.dataframe(
                get_frequent_products(df, selected_user_id),
                use_container_width=True,
            )

        with right_col:
            st.markdown("### Most Recent Historical Products")
            st.dataframe(
                get_recent_products(df, selected_user_id),
                use_container_width=True,
            )

        if enable_llm:
            st.subheader("Local LLM Customer Behaviour Analysis")

            with st.spinner(
                f"Generating local LLM analysis with {llm_model_name}..."
            ):
                try:
                    analysis = analyse_customer_with_local_llm(
                        user_id=selected_user_id,
                        top_k=top_k,
                        model_name=llm_model_name,
                    )

                    st.markdown(analysis)

                except Exception as exc:
                    st.error(
                        "Local LLM analysis failed. Check that Ollama is installed, "
                        "running, and that the selected model has been downloaded."
                    )
                    st.exception(exc)

else:
    st.info("Select a user and click **Generate Recommendations**.")

    st.markdown(
        """
        To enable local LLM explanations, install Ollama and download a lightweight model:

        ```powershell
        ollama pull llama3.2:3b
        ```

        Then tick **Enable local LLM analysis** in the sidebar.
        """
    )
