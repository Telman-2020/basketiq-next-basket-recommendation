import joblib
import optuna
import pandas as pd

from imblearn.under_sampling import RandomUnderSampler

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.basketiq.config import TRAINING_DATA_FILE, PROCESSED_DATA_DIR


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

MODEL_NAMES = [
    "logistic_regression",
    "knn",
    "linear_svm",
    "decision_tree",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
]


def build_model(model_name: str, params: dict | None = None):
    """
    Build a model using either default parameters or Optuna-selected parameters.

    LinearSVC is wrapped with CalibratedClassifierCV so it can provide
    probabilities for soft voting and ranking-style evaluation.
    """
    params = params or {}

    if model_name == "logistic_regression":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=params.get("C", 1.0),
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )

    if model_name == "knn":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    KNeighborsClassifier(
                        n_neighbors=params.get("n_neighbors", 25),
                        weights=params.get("weights", "uniform"),
                    ),
                ),
            ]
        )

    if model_name == "linear_svm":
        base_model = LinearSVC(
            C=params.get("C", 1.0),
            class_weight="balanced",
            random_state=42,
            max_iter=5000,
        )

        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    CalibratedClassifierCV(
                        estimator=base_model,
                        cv=3,
                    ),
                ),
            ]
        )

    if model_name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=params.get("max_depth", 12),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            class_weight="balanced",
            random_state=42,
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 16),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )

    if model_name == "xgboost":
        return XGBClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            min_child_weight=params.get("min_child_weight", 1.0),
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        )

    if model_name == "lightgbm":
        return LGBMClassifier(
            n_estimators=params.get("n_estimators", 300),
            num_leaves=params.get("num_leaves", 64),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", -1),
            min_child_samples=params.get("min_child_samples", 20),
            subsample=params.get("subsample", 1.0),
            colsample_bytree=params.get("colsample_bytree", 1.0),
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )

    if model_name == "catboost":
        return CatBoostClassifier(
            iterations=params.get("iterations", 300),
            depth=params.get("depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            l2_leaf_reg=params.get("l2_leaf_reg", 3.0),
            loss_function="Logloss",
            verbose=False,
            random_state=42,
        )

    raise ValueError(f"Unknown model name: {model_name}")


def get_prediction_scores(model, X_test):
    """
    Return class-1 probabilities or decision scores.

    Hard-voting models do not provide probability scores, so class predictions
    are used as fallback scores for ROC-AUC / PR-AUC.
    """
    try:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X_test)[:, 1]
    except AttributeError:
        pass

    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)

    return model.predict(X_test)


def precision_recall_at_k(df_eval: pd.DataFrame, k: int) -> tuple[float, float]:
    """
    Calculate average Precision@K and Recall@K across users.

    Precision@K:
        Of the top K recommended products, how many were actually reordered?

    Recall@K:
        Of all actually reordered products for the user, how many appeared in the top K?
    """
    precision_values = []
    recall_values = []

    for _, user_df in df_eval.groupby("user_id"):
        top_k = user_df.sort_values("score", ascending=False).head(k)

        true_positives_at_k = top_k["actual"].sum()
        actual_positives = user_df["actual"].sum()

        precision_values.append(true_positives_at_k / k)

        if actual_positives > 0:
            recall_values.append(true_positives_at_k / actual_positives)

    mean_precision_at_k = (
        sum(precision_values) / len(precision_values)
        if precision_values
        else 0.0
    )
    mean_recall_at_k = (
        sum(recall_values) / len(recall_values)
        if recall_values
        else 0.0
    )

    return mean_precision_at_k, mean_recall_at_k


def evaluate_model(
    model_name: str,
    model,
    X_test,
    y_test,
    test_user_ids,
) -> dict:
    y_pred = model.predict(X_test)
    y_score = get_prediction_scores(model, X_test)

    eval_df = pd.DataFrame(
        {
            "user_id": test_user_ids.values,
            "actual": y_test.values,
            "score": y_score,
        }
    )

    precision_at_5, recall_at_5 = precision_recall_at_k(eval_df, k=5)
    precision_at_10, recall_at_10 = precision_recall_at_k(eval_df, k=10)

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "pr_auc": average_precision_score(y_test, y_score),
        "precision_at_5": precision_at_5,
        "recall_at_5": recall_at_5,
        "precision_at_10": precision_at_10,
        "recall_at_10": recall_at_10,
    }


def objective_factory(model_name, X_train, y_train, X_valid, y_valid):
    def objective(trial):
        if model_name == "logistic_regression":
            params = {
                "C": trial.suggest_float("C", 0.001, 10.0, log=True),
            }

        elif model_name == "knn":
            params = {
                "n_neighbors": trial.suggest_int("n_neighbors", 5, 50),
                "weights": trial.suggest_categorical(
                    "weights",
                    ["uniform", "distance"],
                ),
            }

        elif model_name == "linear_svm":
            params = {
                "C": trial.suggest_float("C", 0.001, 10.0, log=True),
            }

        elif model_name == "decision_tree":
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
            }

        elif model_name == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                "max_depth": trial.suggest_int("max_depth", 5, 25),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            }

        elif model_name == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.2,
                    log=True,
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree",
                    0.6,
                    1.0,
                ),
                "min_child_weight": trial.suggest_float(
                    "min_child_weight",
                    1.0,
                    10.0,
                ),
            }

        elif model_name == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "num_leaves": trial.suggest_int("num_leaves", 16, 128),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.2,
                    log=True,
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree",
                    0.6,
                    1.0,
                ),
            }

        elif model_name == "catboost":
            params = {
                "iterations": trial.suggest_int("iterations", 100, 500),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.2,
                    log=True,
                ),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            }

        else:
            raise ValueError(f"Unknown model name: {model_name}")

        model = build_model(model_name, params)
        model.fit(X_train, y_train)

        scores = get_prediction_scores(model, X_valid)

        return average_precision_score(y_valid, scores)

    return objective


def run_bayesian_optimization_with_voting(n_trials: int = 20):
    print("Loading training data...")
    df = pd.read_parquet(TRAINING_DATA_FILE)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)
    user_ids = df["user_id"]

    print(f"Rows: {len(df):,}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Positive class rate: {y.mean():.4f}")

    X_train, X_test, y_train, y_test, user_ids_train, user_ids_test = train_test_split(
        X,
        y,
        user_ids,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train_inner, X_valid, y_train_inner, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    sampler = RandomUnderSampler(
        sampling_strategy=0.3,
        random_state=42,
    )

    X_train_inner_balanced, y_train_inner_balanced = sampler.fit_resample(
        X_train_inner,
        y_train_inner,
    )

    print("\nInner training class distribution after undersampling:")
    print(y_train_inner_balanced.value_counts(normalize=True))

    best_params_by_model = {}
    optimisation_results = []

    for model_name in MODEL_NAMES:
        print("\n" + "=" * 80)
        print(f"Bayesian optimisation for: {model_name}")
        print("=" * 80)

        study = optuna.create_study(direction="maximize")

        study.optimize(
            objective_factory(
                model_name=model_name,
                X_train=X_train_inner_balanced,
                y_train=y_train_inner_balanced,
                X_valid=X_valid,
                y_valid=y_valid,
            ),
            n_trials=n_trials,
        )

        best_params_by_model[model_name] = study.best_params

        optimisation_results.append(
            {
                "model": model_name,
                "best_validation_pr_auc": study.best_value,
                "best_params": study.best_params,
            }
        )

        print(f"Best validation PR-AUC: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")

    print("\nBalancing full training set for final model training...")
    X_train_balanced, y_train_balanced = sampler.fit_resample(X_train, y_train)

    print("\nFull training class distribution after undersampling:")
    print(y_train_balanced.value_counts(normalize=True))
    print(f"Balanced training rows: {len(X_train_balanced):,}")

    tuned_models = {}
    final_results = []

    models_dir = PROCESSED_DATA_DIR / "tuned_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\nTraining tuned models on balanced full training data...")

    for model_name in MODEL_NAMES:
        print("\n" + "-" * 80)
        print(f"Training tuned model: {model_name}")
        print("-" * 80)

        model = build_model(model_name, best_params_by_model[model_name])
        model.fit(X_train_balanced, y_train_balanced)

        tuned_models[model_name] = model

        model_path = models_dir / f"tuned_{model_name}.joblib"
        joblib.dump(model, model_path)

        result = evaluate_model(
            model_name,
            model,
            X_test,
            y_test,
            user_ids_test,
        )
        final_results.append(result)

        print(result)
        print(f"Saved model to: {model_path}")

    print("\nCreating soft VotingClassifier from all tuned models...")

    voting_estimators = [
        (model_name, tuned_models[model_name])
        for model_name in MODEL_NAMES
    ]

    voting_classifier = VotingClassifier(
        estimators=voting_estimators,
        voting="soft",
        n_jobs=-1,
    )

    voting_classifier.fit(X_train_balanced, y_train_balanced)

    voting_result = evaluate_model(
        "soft_voting_classifier_all_models",
        voting_classifier,
        X_test,
        y_test,
        user_ids_test,
    )

    final_results.append(voting_result)

    voting_model_path = models_dir / "soft_voting_classifier_all_models.joblib"
    joblib.dump(voting_classifier, voting_model_path)

    print("\nAll-model soft voting classifier result:")
    print(voting_result)
    print(f"Saved voting classifier to: {voting_model_path}")

    print("\nCreating weighted soft VotingClassifier from boosting models only...")

    boosting_model_names = [
        "xgboost",
        "lightgbm",
        "catboost",
    ]

    boosting_estimators = [
        (model_name, tuned_models[model_name])
        for model_name in boosting_model_names
    ]

    boosting_weighted_voting_classifier = VotingClassifier(
        estimators=boosting_estimators,
        voting="soft",
        weights=[3, 2, 2],
        n_jobs=-1,
    )

    boosting_weighted_voting_classifier.fit(
        X_train_balanced,
        y_train_balanced,
    )

    boosting_voting_result = evaluate_model(
        "weighted_soft_voting_boosting_models",
        boosting_weighted_voting_classifier,
        X_test,
        y_test,
        user_ids_test,
    )

    final_results.append(boosting_voting_result)

    boosting_voting_model_path = (
        models_dir / "weighted_soft_voting_boosting_models.joblib"
    )

    joblib.dump(
        boosting_weighted_voting_classifier,
        boosting_voting_model_path,
    )

    print("\nBoosting-only weighted soft voting classifier result:")
    print(boosting_voting_result)
    print(f"Saved boosting voting classifier to: {boosting_voting_model_path}")

    print("\nCreating hard VotingClassifier from all tuned models...")

    hard_voting_estimators = [
        (model_name, tuned_models[model_name])
        for model_name in MODEL_NAMES
    ]

    hard_voting_classifier = VotingClassifier(
        estimators=hard_voting_estimators,
        voting="hard",
        n_jobs=-1,
    )

    hard_voting_classifier.fit(X_train_balanced, y_train_balanced)

    hard_voting_result = evaluate_model(
        "hard_voting_classifier_all_models",
        hard_voting_classifier,
        X_test,
        y_test,
        user_ids_test,
    )

    final_results.append(hard_voting_result)

    hard_voting_model_path = models_dir / "hard_voting_classifier_all_models.joblib"
    joblib.dump(hard_voting_classifier, hard_voting_model_path)

    print("\nAll-model hard voting classifier result:")
    print(hard_voting_result)
    print(f"Saved hard voting classifier to: {hard_voting_model_path}")

    print("\nCreating hard VotingClassifier from boosting models only...")

    hard_boosting_estimators = [
        (model_name, tuned_models[model_name])
        for model_name in boosting_model_names
    ]

    hard_boosting_voting_classifier = VotingClassifier(
        estimators=hard_boosting_estimators,
        voting="hard",
        n_jobs=-1,
    )

    hard_boosting_voting_classifier.fit(
        X_train_balanced,
        y_train_balanced,
    )

    hard_boosting_voting_result = evaluate_model(
        "hard_voting_boosting_models",
        hard_boosting_voting_classifier,
        X_test,
        y_test,
        user_ids_test,
    )

    final_results.append(hard_boosting_voting_result)

    hard_boosting_voting_model_path = (
        models_dir / "hard_voting_boosting_models.joblib"
    )

    joblib.dump(
        hard_boosting_voting_classifier,
        hard_boosting_voting_model_path,
    )

    print("\nBoosting-only hard voting classifier result:")
    print(hard_boosting_voting_result)
    print(f"Saved hard boosting voting classifier to: {hard_boosting_voting_model_path}")

    optimisation_results_df = pd.DataFrame(optimisation_results)
    optimisation_results_path = (
        PROCESSED_DATA_DIR / "bayesian_optimization_results.csv"
    )
    optimisation_results_df.to_csv(optimisation_results_path, index=False)

    final_results_df = pd.DataFrame(final_results).sort_values(
        by="pr_auc",
        ascending=False,
    )

    final_results_path = PROCESSED_DATA_DIR / "tuned_model_comparison.csv"
    final_results_df.to_csv(final_results_path, index=False)

    print("\nFinal tuned model comparison:")
    print(final_results_df)

    print(f"\nSaved optimisation results to: {optimisation_results_path}")
    print(f"Saved final tuned model comparison to: {final_results_path}")


if __name__ == "__main__":
    run_bayesian_optimization_with_voting(n_trials=20)
