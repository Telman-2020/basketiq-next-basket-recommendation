# Model Selection Report

**Author:** Telman Maghrebi  
**Role:** Data Scientist  
**Date:** 11 March 2025

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

| Model                                |   PR-AUC |     F1 |   Precision |   Recall |   Precision@5 |   Recall@5 |   Precision@10 |   Recall@10 |
|:-------------------------------------|---------:|-------:|------------:|---------:|--------------:|-----------:|---------------:|------------:|
| weighted_soft_voting_boosting_models |   0.4146 | 0.4365 |      0.3822 |   0.5088 |        0.1816 |     0.7623 |         0.1144 |      0.9114 |
| lightgbm                             |   0.413  | 0.385  |      0.263  |   0.7183 |        0.1813 |     0.7607 |         0.1142 |      0.9115 |
| xgboost                              |   0.4126 | 0.4295 |      0.4323 |   0.4266 |        0.1817 |     0.7638 |         0.1145 |      0.911  |
| catboost                             |   0.411  | 0.4261 |      0.4341 |   0.4183 |        0.1819 |     0.7637 |         0.1144 |      0.9102 |
| soft_voting_classifier_all_models    |   0.4079 | 0.4244 |      0.3382 |   0.5696 |        0.181  |     0.7601 |         0.1145 |      0.9113 |
| random_forest                        |   0.3997 | 0.3844 |      0.264  |   0.7067 |        0.1798 |     0.7566 |         0.1145 |      0.9114 |
| logistic_regression                  |   0.3977 | 0.3675 |      0.2438 |   0.746  |        0.1787 |     0.7517 |         0.1142 |      0.91   |
| linear_svm                           |   0.3943 | 0.4149 |      0.4118 |   0.4182 |        0.1787 |     0.7518 |         0.1142 |      0.91   |
| decision_tree                        |   0.3757 | 0.3532 |      0.2319 |   0.7407 |        0.1758 |     0.7405 |         0.1133 |      0.9042 |
| knn                                  |   0.3644 | 0.3995 |      0.4197 |   0.3812 |        0.1772 |     0.7488 |         0.1128 |      0.9018 |
| hard_voting_boosting_models          |   0.24   | 0.4307 |      0.421  |   0.4408 |        0.1522 |     0.645  |         0.1012 |      0.8185 |
| hard_voting_classifier_all_models    |   0.237  | 0.4293 |      0.3883 |   0.4799 |        0.1551 |     0.6566 |         0.1027 |      0.8286 |

---

## 6. Champion Model

The champion model selected by highest PR-AUC is **weighted_soft_voting_boosting_models**.

This selection is appropriate because PR-AUC focuses on the positive class in an imbalanced classification problem.

For a recommendation system, the selected model should also perform well on Precision@K and Recall@K, because the business output is a ranked product list.

---

## 7. Champion Model Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.8721 |
| Precision | 0.3822 |
| Recall | 0.5088 |
| F1-score | 0.4365 |
| ROC-AUC | 0.8327 |
| PR-AUC | 0.4146 |
| Precision@5 | 0.1816 |
| Recall@5 | 0.7623 |
| Precision@10 | 0.1144 |
| Recall@10 | 0.9114 |

---

## 8. Recommendation Metric Interpretation

For the champion model:

- **Precision@5 = 0.1816**  
  Approximately 18.16% of the top 5 recommended products were actually reordered.

- **Recall@5 = 0.7623**  
  Approximately 76.23% of the actually reordered products were captured within the top 5 recommendations.

- **Precision@10 = 0.1144**  
  Approximately 11.44% of the top 10 recommended products were actually reordered.

- **Recall@10 = 0.9114**  
  Approximately 91.14% of the actually reordered products were captured within the top 10 recommendations.

High Recall@K is valuable in grocery recommendation because the system should surface products that customers are likely to need again.

Precision@K is also monitored to ensure that the recommendation list remains relevant and does not become too noisy.

---

## 9. Soft Voting vs Hard Voting

Soft voting is more suitable than hard voting for this recommendation task.

Soft voting uses predicted probabilities, such as `P(reordered_next_order = 1)`. These probabilities can be used to rank products.

Hard voting only uses class labels, either `0` or `1`. This removes ranking information and is therefore less suitable for top-K product recommendation.

---

## 10. Final Decision

The final selected model is **weighted_soft_voting_boosting_models**.

This model was selected because it achieved the highest PR-AUC in the generated comparison table and provides probability scores that can be used to rank products for personalised top-5 and top-10 grocery recommendations.
