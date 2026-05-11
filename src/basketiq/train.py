import joblib
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.basketiq.config import TRAINING_DATA_FILE, PROCESSED_DATA_DIR, MODEL_RESULTS_FILE
from src.basketiq.models import get_models


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

    # Log-transformed features for skewed distributions
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

TARGET_COLUMN = "reordered_next_order"


def get_prediction_scores(model, X_test):
    """
    Return probability-like scores for ranking.
    Some models support predict_proba, while LinearSVC uses decision_function.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)

    return model.predict(X_test)


def train_and_compare_models():
    print("Loading training data...")
    df = pd.read_parquet(TRAINING_DATA_FILE)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)

    print(f"Rows: {len(df):,}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Positive class rate: {y.mean():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print("\nClass distribution before balancing:")
    print(y_train.value_counts(normalize=True))

    sampler = RandomUnderSampler(
        sampling_strategy=0.3,  # Undersample to 30% positive class
        random_state=42,
    )

    X_train_balanced, y_train_balanced = sampler.fit_resample(X_train, y_train)

    print("\nClass distribution after undersampling:")
    print(y_train_balanced.value_counts(normalize=True))
    print(f"Balanced training rows: {len(X_train_balanced):,}")

    models = get_models()
    results = []

    models_dir = PROCESSED_DATA_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        print("\n" + "=" * 60)
        print(f"Training model: {model_name}")
        print("=" * 60)

        # Scale only models that benefit from scaling
        if model_name in ["logistic_regression", "knn", "linear_svm", "rbf_svm"]:
            estimator = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", model),
                ]
            )
        else:
            estimator = model

        estimator.fit(X_train_balanced, y_train_balanced)

        y_pred = estimator.predict(X_test)
        y_score = get_prediction_scores(estimator, X_test)

        result = {
            "model": model_name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_score),
            "pr_auc": average_precision_score(y_test, y_score),
        }

        results.append(result)

        model_path = models_dir / f"{model_name}.joblib"
        joblib.dump(estimator, model_path)

        print(result)
        print(f"Saved model to: {model_path}")

    results_df = pd.DataFrame(results).sort_values(
        by="pr_auc",
        ascending=False,
    )

    results_df.to_csv(MODEL_RESULTS_FILE, index=False)

    print("\nFinal model comparison:")
    print(results_df)
    print(f"\nSaved results to: {MODEL_RESULTS_FILE}")


if __name__ == "__main__":
    train_and_compare_models()