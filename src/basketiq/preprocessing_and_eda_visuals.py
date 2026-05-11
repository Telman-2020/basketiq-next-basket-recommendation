from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.basketiq.config import TRAINING_DATA_FILE


REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
EDA_REPORT_FILE = REPORTS_DIR / "preprocessing_and_eda_report.md"

TARGET_COLUMN = "reordered_next_order"

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

CATEGORICAL_COLUMNS = [
    "product_name",
    "aisle_id",
    "department_id",
]


def save_markdown(section: str, mode: str = "a") -> None:
    with open(EDA_REPORT_FILE, mode, encoding="utf-8") as file:
        file.write(section + "\n\n")


def load_training_data() -> pd.DataFrame:
    print("Loading engineered training dataset...")
    return pd.read_parquet(TRAINING_DATA_FILE)


def dataset_and_feature_engineering_summary(df: pd.DataFrame) -> None:
    section = f"""
# BasketIQ EDA Report

## 1. Purpose of This Report

This report analyses the supervised machine learning dataset created for next-basket product recommendation.

The dataset analysed here is **not one of the original raw CSV files**. It is an engineered modelling table created from the raw Instacart datasets.

Each row represents:

```text
one user_id + one product_id
```

The machine learning question is:

```text
Will this customer reorder this product in the next labelled order?
```

The target variable is:

```text
reordered_next_order
```

where:

```text
1 = customer reordered the product in the next labelled order
0 = customer did not reorder the product in the next labelled order
```

---

## 2. Raw Data Sources Used

The final modelling table was created from the following raw datasets:

| Raw Dataset | Purpose |
|---|---|
| `orders.csv` | Contains user order history, order sequence, order timing and evaluation split |
| `order_products__prior.csv` | Contains products purchased in historical prior orders |
| `order_products__train.csv` | Contains products purchased in the labelled next order |
| `products.csv` | Provides product names, aisle IDs and department IDs |
| `aisles.csv` | Provides aisle names |
| `departments.csv` | Provides department names |

The raw data is split across several tables, so feature engineering is required before machine learning can be applied.

---

## 3. How the Main Modelling Table Was Created

The final dataset was created through the following process.

### Step 1: Select labelled training orders

From `orders.csv`, rows where `eval_set = train` were selected.

This identifies each user's labelled next order. These orders are used to create the target variable.

### Step 2: Select historical prior orders

From `orders.csv`, rows where `eval_set = prior` were selected.

These represent the customer's historical shopping behaviour before the labelled train order.

### Step 3: Join prior orders with prior products

`order_products__prior.csv` was joined with the prior orders from `orders.csv` using `order_id`.

This connects each purchased product to the customer who bought it and the order sequence in which it appeared.

### Step 4: Create user-product features

The joined prior history was grouped by `user_id, product_id`.

This creates personalised customer-product behavioural features.

### Step 5: Create user-level features

The prior orders were grouped by `user_id`.

This creates features describing the customer's overall shopping behaviour.

### Step 6: Create product-level features

The joined prior product history was grouped by `product_id`.

This creates features describing product popularity and repeat-purchase behaviour.

### Step 7: Create the target variable

`order_products__train.csv` was joined with the labelled train orders.

If a previously purchased product appears in the user's labelled next order, the target is `reordered_next_order = 1`.

If it does not appear, the target is `reordered_next_order = 0`.

### Step 8: Join all components

The final table joins:

| Component | Join Key |
|---|---|
| User-product features + user features | `user_id` |
| User-product features + product features | `product_id` |
| User-product features + product metadata | `product_id` |
| User-product features + target labels | `user_id`, `product_id` |

The result is one supervised ML table containing customer-product behaviour, product information and the binary target.

---

## 4. Raw Features vs Engineered Features

### Raw Features

Raw features come directly from the original CSV files.

| Raw Feature | Source | Meaning |
|---|---|---|
| `user_id` | `orders.csv` | Customer identifier |
| `order_id` | Multiple files | Order identifier |
| `product_id` | Order-product files, `products.csv` | Product identifier |
| `order_number` | `orders.csv` | Sequence number of the order for a customer |
| `days_since_prior_order` | `orders.csv` | Days since the customer's previous order |
| `add_to_cart_order` | Order-product files | Position where the item was added to basket |
| `reordered` | Order-product files | Whether the product had been ordered before |
| `product_name` | `products.csv` | Product name |
| `aisle_id` | `products.csv` | Product aisle identifier |
| `department_id` | `products.csv` | Product department identifier |

### Engineered Features

Engineered features are created through grouping, aggregation, ratios and joins.

| Engineered Feature | Created From | Product Recommendation Meaning |
|---|---|---|
| `user_product_total_orders` | User-product prior history | How often this customer bought this product |
| `user_product_total_reorders` | User-product prior history | How often this customer repeatedly bought this product |
| `user_product_avg_cart_order` | `add_to_cart_order` | How early this product is usually added to the basket |
| `user_product_last_order_number` | `order_number` | The last order number where the customer bought this product |
| `user_total_orders` | User prior order history | How much purchase history the customer has |
| `user_avg_days_since_prior_order` | `days_since_prior_order` | How frequently the customer usually shops |
| `product_total_purchases` | Product prior history | How popular the product is overall |
| `product_total_reorders` | Product prior history | How often the product is reordered overall |
| `product_reorder_rate` | Product reorders / product purchases | Whether the product behaves like a repeat-purchase item |
| `user_product_reorder_rate` | User-product reorders / user-product purchases | How strongly this customer repeatedly buys this product |
| `orders_since_last_purchase` | User total orders - last product order number | How recently the customer bought this product |

---

## 5. Feature Meaning from a Product Viewpoint

The engineered features describe four important product recommendation signals.

### Frequency

Frequency captures how often the customer has bought a product.

Relevant features:

- `user_product_total_orders`
- `user_product_total_reorders`

A product bought many times by the same customer is likely to be personally relevant.

### Recency

Recency captures how recently the customer bought a product.

Relevant features:

- `user_product_last_order_number`
- `orders_since_last_purchase`

A product bought recently may have a higher chance of appearing again in the next basket.

### Product Popularity

Product popularity captures how common the product is across customers.

Relevant features:

- `product_total_purchases`
- `product_total_reorders`

Popular products can be useful baseline recommendations, but they need to be combined with personal behaviour.

### Product Repeatability

Product repeatability captures whether a product is naturally bought again and again.

Relevant features:

- `product_reorder_rate`
- `user_product_reorder_rate`

High-repeat products are often grocery staples or habitual purchases.

---

## 6. Dataset Size

The processed supervised learning dataset contains:

- Rows: {df.shape[0]:,}
- Columns: {df.shape[1]:,}

Each row represents one customer-product pair.
"""
    save_markdown(section, mode="w")


def missing_value_check(df: pd.DataFrame) -> None:
    missing = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_percentage": (df.isna().mean().values * 100).round(2),
        }
    )

    section = f"""
## 7. Missing Value Check

The table below shows missing values in the engineered modelling dataset.

{missing.to_markdown(index=False)}

### Observation

Missing values should be reviewed carefully because some missing values may have business meaning.

For example, after target creation, a missing match between historical user-product pairs and the labelled next order means the product was **not reordered** in the next basket.

However, missing values in raw product metadata or text fields should not be blindly replaced with zero.
"""
    save_markdown(section)


def numerical_summary(df: pd.DataFrame) -> None:
    summary = df[FEATURE_COLUMNS + [TARGET_COLUMN]].describe().T

    section = f"""
## 8. Numerical Feature Summary

The table below shows the count, mean, standard deviation, minimum, quartiles and maximum for each numerical feature.

{summary.to_markdown()}

### Observation

Count-based features such as `product_total_purchases`, `product_total_reorders`, and `user_product_total_orders` are expected to be right-skewed.

Rate-based features such as `product_reorder_rate` and `user_product_reorder_rate` are bounded between 0 and 1.

Recency-based features such as `orders_since_last_purchase` help capture how recently the customer bought a product.
"""
    save_markdown(section)


def categorical_summary(df: pd.DataFrame) -> None:
    section = """
## 9. Categorical Feature Summary

The categorical variables mainly describe product metadata.

"""

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            unique_count = df[col].nunique(dropna=True)
            top_values = df[col].value_counts(dropna=False).head(10)

            section += f"""
### {col}

- Unique categories: {unique_count:,}

Top categories:

{top_values.to_markdown()}
"""

    section += """
### Observation

Product-related categorical features provide business context.

`product_name` is a high-cardinality categorical feature, which means it has many unique values.

In the baseline model, product names are used for interpretation and reporting rather than direct modelling.
"""
    save_markdown(section)


def target_distribution(df: pd.DataFrame) -> None:
    target_counts = df[TARGET_COLUMN].value_counts().sort_index()
    target_rate = df[TARGET_COLUMN].mean()

    plt.figure(figsize=(6, 4))
    target_counts.plot(kind="bar")
    plt.title("Target Distribution")
    plt.xlabel("Reordered in Next Order")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png")
    plt.close()

    section = f"""
## 10. Target Distribution

Target counts:

{target_counts.to_markdown()}

Positive class rate: {target_rate:.4f}

![Target Distribution](figures/target_distribution.png)

### Observation

The target is expected to be imbalanced because most historical customer-product pairs are not reordered in the next basket.

This means accuracy alone is not enough. Precision, recall, F1-score, ROC-AUC, PR-AUC and Precision@K are more useful evaluation metrics.
"""
    save_markdown(section)


def histogram_plots(df: pd.DataFrame) -> None:
    section = """
## 11. Histogram Analysis

Histograms were generated for each numerical feature.

"""

    for col in FEATURE_COLUMNS:
        plt.figure(figsize=(7, 4))
        df[col].dropna().hist(bins=50)
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()

        filename = f"hist_{col}.png"
        plt.savefig(FIGURES_DIR / filename)
        plt.close()

        section += f"""
### {col}

![Histogram of {col}](figures/{filename})

Observation: The histogram shows the distribution of `{col}` and helps identify skewness, concentration around common values and possible outliers.
"""

    save_markdown(section)


def boxplots(df: pd.DataFrame) -> None:
    section = """
## 12. Boxplot Analysis

Boxplots were generated to identify outliers and spread in numerical features.

"""

    for col in FEATURE_COLUMNS:
        plt.figure(figsize=(7, 4))
        plt.boxplot(df[col].dropna(), vert=False)
        plt.title(f"Boxplot of {col}")
        plt.xlabel(col)
        plt.tight_layout()

        filename = f"box_{col}.png"
        plt.savefig(FIGURES_DIR / filename)
        plt.close()

        section += f"""
### {col}

![Boxplot of {col}](figures/{filename})

Observation: The boxplot helps identify whether `{col}` contains extreme values that may influence models, especially distance-based or linear models.
"""

    save_markdown(section)


def scatter_plots(df: pd.DataFrame, max_rows: int = 10000) -> None:
    section = """
## 13. Scatter Plot Analysis

Scatter plots were generated for selected feature pairs using a sample of the dataset to keep the plots readable.

"""

    sample_df = df.sample(n=min(max_rows, len(df)), random_state=42)

    feature_pairs = [
        ("user_product_total_orders", "user_product_reorder_rate"),
        ("product_total_purchases", "product_reorder_rate"),
        ("orders_since_last_purchase", "user_product_reorder_rate"),
        ("user_total_orders", "user_product_total_orders"),
        ("product_total_reorders", "product_reorder_rate"),
    ]

    for x_col, y_col in feature_pairs:
        plt.figure(figsize=(7, 4))
        plt.scatter(sample_df[x_col], sample_df[y_col], alpha=0.3)
        plt.title(f"{x_col} vs {y_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.tight_layout()

        filename = f"scatter_{x_col}_vs_{y_col}.png"
        plt.savefig(FIGURES_DIR / filename)
        plt.close()

        section += f"""
### {x_col} vs {y_col}

![Scatter plot](figures/{filename})

Observation: This plot helps assess whether `{x_col}` and `{y_col}` show a visible relationship, clustering pattern or non-linear behaviour.
"""

    save_markdown(section)


def correlation_analysis(df: pd.DataFrame) -> None:
    corr = df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr()

    plt.figure(figsize=(12, 9))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_matrix.png")
    plt.close()

    target_corr = corr[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values(ascending=False)

    section = f"""
## 14. Correlation and Relationship Analysis

The correlation matrix below shows pairwise linear relationships between numerical features and the target.

![Correlation Matrix](figures/correlation_matrix.png)

### Correlation with Target

{target_corr.to_markdown()}

### Observation

Features with higher positive correlation to the target may be useful predictors of next-basket reorder behaviour.

Correlation only captures linear relationships. Tree-based models such as Random Forest, XGBoost, LightGBM and CatBoost can capture non-linear interactions.
"""
    save_markdown(section)


def feature_target_relationship(df: pd.DataFrame) -> None:
    grouped = df.groupby(TARGET_COLUMN)[FEATURE_COLUMNS].mean().T

    section = f"""
## 15. Feature vs Target Analysis

The mean value of each feature was compared between target classes.

{grouped.to_markdown()}

### Observation

This comparison shows how customer-product behaviour differs between products that were reordered and products that were not reordered.

Products that were reordered are expected to have stronger customer-product history, higher reorder rates and stronger recency signals.
"""
    save_markdown(section)


def additional_analysis(df: pd.DataFrame) -> None:
    section = """
## 16. Additional Useful Analysis

### Top Products by Frequency

"""

    if "product_name" in df.columns:
        top_products = df["product_name"].value_counts().head(20)
        section += top_products.to_markdown()

    section += """

### Business Observations

- Frequently purchased products can dominate recommendations, so the model should balance popularity with personalisation.
- Recency and reorder history are important because grocery shopping contains repeated habits.
- Product-level reorder rate helps identify staple products.
- User-product interaction features are more personalised than global product popularity features.
- The class imbalance suggests that PR-AUC and Precision@K are important metrics for evaluating recommendation quality.
"""
    save_markdown(section)


def run_eda() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_training_data()

    dataset_and_feature_engineering_summary(df)
    missing_value_check(df)
    numerical_summary(df)
    categorical_summary(df)
    target_distribution(df)
    histogram_plots(df)
    boxplots(df)
    scatter_plots(df)
    correlation_analysis(df)
    feature_target_relationship(df)
    additional_analysis(df)

    print(f"EDA complete. Report saved to: {EDA_REPORT_FILE}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    run_eda()
