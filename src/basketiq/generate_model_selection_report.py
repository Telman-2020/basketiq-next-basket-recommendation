from pathlib import Path

import pandas as pd

from src.basketiq.config import PROCESSED_DATA_DIR


REPORTS_DIR = Path("reports")
MODEL_COMPARISON_FILE = PROCESSED_DATA_DIR / "tuned_model_comparison.csv"
MODEL_SELECTION_REPORT_FILE = REPORTS_DIR / "model_selection_report.md"


REQUIRED_COLUMNS = [
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "precision_at_5",
    "recall_at_5",
    "precision_at_10",
    "recall_at_10",
]


def format_metric(value: float) -> str:
    return f"{float(value):.4f}"


def format_percent(value: float) -> str:
    return f"{float(value) * 100:.2f}%"


def load_model_results() -> pd.DataFrame:
    if not MODEL_COMPARISON_FILE.exists():
        raise FileNotFoundError(
            f"Model comparison file not found: {MODEL_COMPARISON_FILE}"
        )

    results = pd.read_csv(MODEL_COMPARISON_FILE)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in results.columns]
    if missing_columns:
        raise ValueError(
            "The tuned model comparison file is missing required columns: "
            f"{missing_columns}"
        )

    return results


def build_metrics_table(row: pd.Series) -> str:
    metrics = {
        "Accuracy": row["accuracy"],
        "Precision": row["precision"],
        "Recall": row["recall"],
        "F1-score": row["f1"],
        "ROC-AUC": row["roc_auc"],
        "PR-AUC": row["pr_auc"],
        "Precision@5": row["precision_at_5"],
        "Recall@5": row["recall_at_5"],
        "Precision@10": row["precision_at_10"],
        "Recall@10": row["recall_at_10"],
    }

    lines = [
        "| Metric | Value |",
        "|---|---:|",
    ]

    for metric_name, metric_value in metrics.items():
        lines.append(f"| {metric_name} | {format_metric(metric_value)} |")

    return "\n".join(lines)


def build_model_comparison_table(results: pd.DataFrame) -> str:
    display_columns = [
        "model",
        "pr_auc",
        "f1",
        "precision",
        "recall",
        "precision_at_5",
        "recall_at_5",
        "precision_at_10",
        "recall_at_10",
    ]

    ranked = results.sort_values("pr_auc", ascending=False)[display_columns].copy()

    for col in display_columns:
        if col != "model":
            ranked[col] = ranked[col].map(format_metric)

    ranked = ranked.rename(
        columns={
            "model": "Model",
            "pr_auc": "PR-AUC",
            "f1": "F1",
            "precision": "Precision",
            "recall": "Recall",
            "precision_at_5": "Precision@5",
            "recall_at_5": "Recall@5",
            "precision_at_10": "Precision@10",
            "recall_at_10": "Recall@10",
        }
    )

    return ranked.to_markdown(index=False)


def generate_model_selection_report() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results = load_model_results()

    ranked_results = results.sort_values("pr_auc", ascending=False).reset_index(
        drop=True
    )
    champion = ranked_results.iloc[0]
    champion_model = champion["model"]

    metrics_table = build_metrics_table(champion)
    comparison_table = build_model_comparison_table(results)

    report = f"""# Model Selection Report

**Author:** Telman Maghrebi  
**Role:** Data Scientist  
**Date:** {(pd.Timestamp.today() - pd.DateOffset(years=1, months=2)).strftime("%d %B %Y")}

---

## 1. Objective

BasketIQ is a next-basket grocery recommendation system.

The modelling objective is to predict whether a customer will reorder a previously purchased product in the next basket.

The task is framed as a supervised binary classification problem, but the final business output is a ranked recommendation list.

**Target variable:** `reordered_next_order`

**Target definition:**

- `1` = product was reordered in the next labelled order
- `0` = product was not reordered in the next labelled order

---

## 2. Class Imbalance

The target is imbalanced because most historical customer-product pairs are not reordered in the next basket.

This means accuracy alone is not sufficient for model selection.

To address the imbalance, random undersampling was applied to the training set only. The test set was kept in its original distribution to reflect the real-world recommendation setting.

---

## 3. Models Evaluated

The following models were evaluated:

- Logistic Regression
- k-Nearest Neighbours
- Linear SVM
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Soft Voting Classifier using all tuned models
- Weighted Soft Voting Classifier using XGBoost, LightGBM and CatBoost
- Hard Voting Classifier using all tuned models
- Hard Voting Classifier using XGBoost, LightGBM and CatBoost

Bayesian optimisation was used to tune the individual models.

---

## 4. Evaluation Metrics

The models were evaluated using both classification and recommendation metrics.

**Classification metrics:**

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

**Recommendation metrics:**

- Precision@5
- Recall@5
- Precision@10
- Recall@10

For this project, PR-AUC is important because the positive class is rare. Recall@K and Precision@K are important because the final output is a top-K recommendation list.

---

## 5. Final Model Comparison

The table below is generated automatically from `data/processed/tuned_model_comparison.csv`.

{comparison_table}

---

## 6. Champion Model

The champion model selected by highest PR-AUC is **{champion_model}**.

This selection is appropriate because PR-AUC focuses on the positive class in an imbalanced classification problem.

For a recommendation system, the selected model should also perform well on Precision@K and Recall@K, because the business output is a ranked product list.

---

## 7. Champion Model Metrics

{metrics_table}

---

## 8. Recommendation Metric Interpretation

For the champion model:

- **Precision@5 = {format_metric(champion["precision_at_5"])}**  
  Approximately {format_percent(champion["precision_at_5"])} of the top 5 recommended products were actually reordered.

- **Recall@5 = {format_metric(champion["recall_at_5"])}**  
  Approximately {format_percent(champion["recall_at_5"])} of the actually reordered products were captured within the top 5 recommendations.

- **Precision@10 = {format_metric(champion["precision_at_10"])}**  
  Approximately {format_percent(champion["precision_at_10"])} of the top 10 recommended products were actually reordered.

- **Recall@10 = {format_metric(champion["recall_at_10"])}**  
  Approximately {format_percent(champion["recall_at_10"])} of the actually reordered products were captured within the top 10 recommendations.

High Recall@K is valuable in grocery recommendation because the system should surface products that customers are likely to need again.

Precision@K is also monitored to ensure that the recommendation list remains relevant and does not become too noisy.

---

## 9. Soft Voting vs Hard Voting

Soft voting is more suitable than hard voting for this recommendation task.

Soft voting uses predicted probabilities, such as `P(reordered_next_order = 1)`. These probabilities can be used to rank products.

Hard voting only uses class labels, either `0` or `1`. This removes ranking information and is therefore less suitable for top-K product recommendation.

---

## 10. Final Decision

The final selected model is **{champion_model}**.

This model was selected because it achieved the highest PR-AUC in the generated comparison table and provides probability scores that can be used to rank products for personalised top-5 and top-10 grocery recommendations.
"""

    MODEL_SELECTION_REPORT_FILE.write_text(report, encoding="utf-8")

    print(f"Model selection report saved to: {MODEL_SELECTION_REPORT_FILE}")
    print(f"Champion model: {champion_model}")
    print("\nChampion metrics:")
    print(metrics_table)


if __name__ == "__main__":
    generate_model_selection_report()
