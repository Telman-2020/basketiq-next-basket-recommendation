
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

- Rows: 326,778
- Columns: 40

Each row represents one customer-product pair.



## 7. Missing Value Check

The table below shows missing values in the engineered modelling dataset.

| column                               |   missing_count |   missing_percentage |
|:-------------------------------------|----------------:|---------------------:|
| user_id                              |               0 |                    0 |
| product_id                           |               0 |                    0 |
| user_product_total_orders            |               0 |                    0 |
| user_product_total_reorders          |               0 |                    0 |
| user_product_avg_cart_order          |               0 |                    0 |
| user_product_last_order_number       |               0 |                    0 |
| user_total_orders                    |               0 |                    0 |
| user_avg_days_since_prior_order      |               0 |                    0 |
| user_avg_basket_size                 |               0 |                    0 |
| user_max_basket_size                 |               0 |                    0 |
| user_total_products                  |               0 |                    0 |
| user_total_reorders                  |               0 |                    0 |
| user_reorder_rate                    |               0 |                    0 |
| product_total_purchases              |               0 |                    0 |
| product_total_reorders               |               0 |                    0 |
| product_avg_add_to_cart_order        |               0 |                    0 |
| product_reorder_rate                 |               0 |                    0 |
| product_first_cart_count             |               0 |                    0 |
| product_first_cart_rate              |               0 |                    0 |
| product_name                         |               0 |                    0 |
| aisle_id                             |               0 |                    0 |
| department_id                        |               0 |                    0 |
| reordered_next_order                 |               0 |                    0 |
| user_product_reorder_rate            |               0 |                    0 |
| orders_since_last_purchase           |               0 |                    0 |
| user_product_purchase_share          |               0 |                    0 |
| log1p_user_product_total_orders      |               0 |                    0 |
| log1p_user_product_total_reorders    |               0 |                    0 |
| log1p_user_product_avg_cart_order    |               0 |                    0 |
| log1p_user_product_last_order_number |               0 |                    0 |
| log1p_user_total_orders              |               0 |                    0 |
| log1p_user_avg_basket_size           |               0 |                    0 |
| log1p_user_max_basket_size           |               0 |                    0 |
| log1p_user_total_products            |               0 |                    0 |
| log1p_user_total_reorders            |               0 |                    0 |
| log1p_product_total_purchases        |               0 |                    0 |
| log1p_product_total_reorders         |               0 |                    0 |
| log1p_product_avg_add_to_cart_order  |               0 |                    0 |
| log1p_product_first_cart_count       |               0 |                    0 |
| log1p_orders_since_last_purchase     |               0 |                    0 |

### Observation

Missing values should be reviewed carefully because some missing values may have business meaning.

For example, after target creation, a missing match between historical user-product pairs and the labelled next order means the product was **not reordered** in the next basket.

However, missing values in raw product metadata or text fields should not be blindly replaced with zero.



## 8. Numerical Feature Summary

The table below shows the count, mean, standard deviation, minimum, quartiles and maximum for each numerical feature.

|                                      |   count |        mean |          std |      min |        25% |         50% |        75% |          max |
|:-------------------------------------|--------:|------------:|-------------:|---------:|-----------:|------------:|-----------:|-------------:|
| user_product_total_orders            |  326778 |   2.46183   |    3.70084   | 1        |  1         |   1         |   2        |    94        |
| user_product_total_reorders          |  326778 |   1.46183   |    3.70084   | 0        |  0         |   0         |   1        |    93        |
| user_product_avg_cart_order          |  326778 |   9.1744    |    6.93443   | 1        |  4         |   7.5       |  12        |    83        |
| user_product_last_order_number       |  326778 |  16.3174    |   17.9881    | 1        |  4         |  10         |  22        |    99        |
| user_total_orders                    |  326778 |  26.1732    |   23.1486    | 3        |  9         |  18         |  37        |    99        |
| user_avg_days_since_prior_order      |  326778 |  12.5249    |    6.584     | 0.333333 |  7.26087   |  11.3       |  16.875    |    30        |
| user_avg_basket_size                 |  326778 |  12.6195    |    6.2818    | 1        |  8.16364   |  11.4717    |  15.7143   |    61        |
| user_max_basket_size                 |  326778 |  24.1362    |   11.2861    | 1        | 16         |  22         |  30        |    84        |
| user_total_products                  |  326778 | 318.922     |  327.439     | 3        | 99         | 207         | 429        |  2256        |
| user_total_reorders                  |  326778 | 203.414     |  264.495     | 0        | 36         | 104         | 263        |  1944        |
| user_reorder_rate                    |  326778 |   0.503208  |    0.196905  | 0        |  0.36036   |   0.519397  |   0.656331 |     0.972222 |
| product_total_purchases              |  326778 | 557.317     | 1401.28      | 1        | 26         | 107         | 410        | 12098        |
| product_total_reorders               |  326778 | 394.783     | 1135.89      | 0        | 10         |  55         | 244        | 10287        |
| product_avg_add_to_cart_order        |  326778 |   8.80853   |    2.671     | 1        |  7.4       |   8.7619    |   9.94626  |    83        |
| product_reorder_rate                 |  326778 |   0.503709  |    0.217574  | 0        |  0.380952  |   0.55      |   0.664625 |     0.97561  |
| product_first_cart_count             |  326778 |  66.7275    |  268.88      | 0        |  2         |   8         |  36        |  2939        |
| product_first_cart_rate              |  326778 |   0.0896626 |    0.0865715 | 0        |  0.0444874 |   0.0744243 |   0.113208 |     1        |
| user_product_reorder_rate            |  326778 |   0.265582  |    0.340035  | 0        |  0         |   0         |   0.5      |     0.989362 |
| orders_since_last_purchase           |  326778 |   9.85583   |   13.9285    | 0        |  1         |   5         |  12        |    98        |
| user_product_purchase_share          |  326778 |   0.151966  |    0.163115  | 0.010101 |  0.0434783 |   0.0909091 |   0.2      |     1        |
| log1p_user_product_total_orders      |  326778 |   1.03129   |    0.537363  | 0.693147 |  0.693147  |   0.693147  |   1.09861  |     4.55388  |
| log1p_user_product_total_reorders    |  326778 |   0.502364  |    0.742142  | 0        |  0         |   0         |   0.693147 |     4.54329  |
| log1p_user_product_avg_cart_order    |  326778 |   2.10481   |    0.670867  | 0.693147 |  1.60944   |   2.14007   |   2.56495  |     4.43082  |
| log1p_user_product_last_order_number |  326778 |   2.39123   |    0.969222  | 0.693147 |  1.60944   |   2.3979    |   3.13549  |     4.60517  |
| log1p_user_total_orders              |  326778 |   2.95677   |    0.849003  | 1.38629  |  2.30259   |   2.94444   |   3.63759  |     4.60517  |
| log1p_user_avg_basket_size           |  326778 |   2.51049   |    0.455968  | 0.693147 |  2.21524   |   2.52346   |   2.81626  |     4.12713  |
| log1p_user_max_basket_size           |  326778 |   3.1227    |    0.465181  | 0.693147 |  2.83321   |   3.13549   |   3.43399  |     4.44265  |
| log1p_user_total_products            |  326778 |   5.3014    |    1.0128    | 1.38629  |  4.60517   |   5.33754   |   6.06379  |     7.72179  |
| log1p_user_total_reorders            |  326778 |   4.52963   |    1.40514   | 0        |  3.61092   |   4.65396   |   5.57595  |     7.57302  |
| log1p_product_total_purchases        |  326778 |   4.67394   |    1.91972   | 0.693147 |  3.29584   |   4.68213   |   6.01859  |     9.40088  |
| log1p_product_total_reorders         |  326778 |   3.9806    |    2.16204   | 0        |  2.3979    |   4.02535   |   5.50126  |     9.23873  |
| log1p_product_avg_add_to_cart_order  |  326778 |   2.24794   |    0.271704  | 0.693147 |  2.12823   |   2.27849   |   2.393    |     4.43082  |
| log1p_product_first_cart_count       |  326778 |   2.40366   |    1.75198   | 0        |  1.09861   |   2.19722   |   3.61092  |     7.98616  |
| log1p_orders_since_last_purchase     |  326778 |   1.73298   |    1.15878   | 0        |  0.693147  |   1.79176   |   2.56495  |     4.59512  |
| reordered_next_order                 |  326778 |   0.0974117 |    0.296518  | 0        |  0         |   0         |   0        |     1        |

### Observation

Count-based features such as `product_total_purchases`, `product_total_reorders`, and `user_product_total_orders` are expected to be right-skewed.

Rate-based features such as `product_reorder_rate` and `user_product_reorder_rate` are bounded between 0 and 1.

Recency-based features such as `orders_since_last_purchase` help capture how recently the customer bought a product.



## 9. Categorical Feature Summary

The categorical variables mainly describe product metadata.


### product_name

- Unique categories: 29,000

Top categories:

| product_name           |   count |
|:-----------------------|--------:|
| Banana                 |    1811 |
| Bag of Organic Bananas |    1549 |
| Organic Strawberries   |    1458 |
| Organic Baby Spinach   |    1314 |
| Large Lemon            |    1142 |
| Limes                  |    1083 |
| Strawberries           |    1075 |
| Organic Avocado        |    1033 |
| Organic Hass Avocado   |    1015 |
| Organic Yellow Onion   |     871 |

### aisle_id

- Unique categories: 134

Top categories:

|   aisle_id |   count |
|-----------:|--------:|
|         83 |   34324 |
|         24 |   25378 |
|        123 |   15647 |
|        120 |   11362 |
|         21 |   10097 |
|        107 |    7096 |
|         37 |    6218 |
|        116 |    5870 |
|         17 |    5759 |
|        115 |    5402 |

### department_id

- Unique categories: 21

Top categories:

|   department_id |   count |
|----------------:|--------:|
|               4 |   81559 |
|              16 |   43956 |
|              13 |   30569 |
|              19 |   29481 |
|               1 |   25099 |
|               7 |   22687 |
|              15 |   14470 |
|               9 |   11528 |
|               3 |   10906 |
|              17 |   10609 |

### Observation

Product-related categorical features provide business context.

`product_name` is a high-cardinality categorical feature, which means it has many unique values.

In the baseline model, product names are used for interpretation and reporting rather than direct modelling.



## 10. Target Distribution

Target counts:

|   reordered_next_order |   count |
|-----------------------:|--------:|
|                      0 |  294946 |
|                      1 |   31832 |

Positive class rate: 0.0974

![Target Distribution](figures/target_distribution.png)

### Observation

The target is expected to be imbalanced because most historical customer-product pairs are not reordered in the next basket.

This means accuracy alone is not enough. Precision, recall, F1-score, ROC-AUC, PR-AUC and Precision@K are more useful evaluation metrics.



## 11. Histogram Analysis

Histograms were generated for each numerical feature.


### user_product_total_orders

![Histogram of user_product_total_orders](figures/hist_user_product_total_orders.png)

Observation: The histogram shows the distribution of `user_product_total_orders` and helps identify skewness, concentration around common values and possible outliers.

### user_product_total_reorders

![Histogram of user_product_total_reorders](figures/hist_user_product_total_reorders.png)

Observation: The histogram shows the distribution of `user_product_total_reorders` and helps identify skewness, concentration around common values and possible outliers.

### user_product_avg_cart_order

![Histogram of user_product_avg_cart_order](figures/hist_user_product_avg_cart_order.png)

Observation: The histogram shows the distribution of `user_product_avg_cart_order` and helps identify skewness, concentration around common values and possible outliers.

### user_product_last_order_number

![Histogram of user_product_last_order_number](figures/hist_user_product_last_order_number.png)

Observation: The histogram shows the distribution of `user_product_last_order_number` and helps identify skewness, concentration around common values and possible outliers.

### user_total_orders

![Histogram of user_total_orders](figures/hist_user_total_orders.png)

Observation: The histogram shows the distribution of `user_total_orders` and helps identify skewness, concentration around common values and possible outliers.

### user_avg_days_since_prior_order

![Histogram of user_avg_days_since_prior_order](figures/hist_user_avg_days_since_prior_order.png)

Observation: The histogram shows the distribution of `user_avg_days_since_prior_order` and helps identify skewness, concentration around common values and possible outliers.

### user_avg_basket_size

![Histogram of user_avg_basket_size](figures/hist_user_avg_basket_size.png)

Observation: The histogram shows the distribution of `user_avg_basket_size` and helps identify skewness, concentration around common values and possible outliers.

### user_max_basket_size

![Histogram of user_max_basket_size](figures/hist_user_max_basket_size.png)

Observation: The histogram shows the distribution of `user_max_basket_size` and helps identify skewness, concentration around common values and possible outliers.

### user_total_products

![Histogram of user_total_products](figures/hist_user_total_products.png)

Observation: The histogram shows the distribution of `user_total_products` and helps identify skewness, concentration around common values and possible outliers.

### user_total_reorders

![Histogram of user_total_reorders](figures/hist_user_total_reorders.png)

Observation: The histogram shows the distribution of `user_total_reorders` and helps identify skewness, concentration around common values and possible outliers.

### user_reorder_rate

![Histogram of user_reorder_rate](figures/hist_user_reorder_rate.png)

Observation: The histogram shows the distribution of `user_reorder_rate` and helps identify skewness, concentration around common values and possible outliers.

### product_total_purchases

![Histogram of product_total_purchases](figures/hist_product_total_purchases.png)

Observation: The histogram shows the distribution of `product_total_purchases` and helps identify skewness, concentration around common values and possible outliers.

### product_total_reorders

![Histogram of product_total_reorders](figures/hist_product_total_reorders.png)

Observation: The histogram shows the distribution of `product_total_reorders` and helps identify skewness, concentration around common values and possible outliers.

### product_avg_add_to_cart_order

![Histogram of product_avg_add_to_cart_order](figures/hist_product_avg_add_to_cart_order.png)

Observation: The histogram shows the distribution of `product_avg_add_to_cart_order` and helps identify skewness, concentration around common values and possible outliers.

### product_reorder_rate

![Histogram of product_reorder_rate](figures/hist_product_reorder_rate.png)

Observation: The histogram shows the distribution of `product_reorder_rate` and helps identify skewness, concentration around common values and possible outliers.

### product_first_cart_count

![Histogram of product_first_cart_count](figures/hist_product_first_cart_count.png)

Observation: The histogram shows the distribution of `product_first_cart_count` and helps identify skewness, concentration around common values and possible outliers.

### product_first_cart_rate

![Histogram of product_first_cart_rate](figures/hist_product_first_cart_rate.png)

Observation: The histogram shows the distribution of `product_first_cart_rate` and helps identify skewness, concentration around common values and possible outliers.

### user_product_reorder_rate

![Histogram of user_product_reorder_rate](figures/hist_user_product_reorder_rate.png)

Observation: The histogram shows the distribution of `user_product_reorder_rate` and helps identify skewness, concentration around common values and possible outliers.

### orders_since_last_purchase

![Histogram of orders_since_last_purchase](figures/hist_orders_since_last_purchase.png)

Observation: The histogram shows the distribution of `orders_since_last_purchase` and helps identify skewness, concentration around common values and possible outliers.

### user_product_purchase_share

![Histogram of user_product_purchase_share](figures/hist_user_product_purchase_share.png)

Observation: The histogram shows the distribution of `user_product_purchase_share` and helps identify skewness, concentration around common values and possible outliers.

### log1p_user_product_total_orders

![Histogram of log1p_user_product_total_orders](figures/hist_log1p_user_product_total_orders.png)

Observation: The histogram shows the distribution of `log1p_user_product_total_orders` and helps identify skewness, concentration around common values and possible outliers.

### log1p_user_product_total_reorders

![Histogram of log1p_user_product_total_reorders](figures/hist_log1p_user_product_total_reorders.png)

Observation: The histogram shows the distribution of `log1p_user_product_total_reorders` and helps identify skewness, concentration around common values and possible outliers.

### log1p_user_product_avg_cart_order

![Histogram of log1p_user_product_avg_cart_order](figures/hist_log1p_user_product_avg_cart_order.png)

Observation: The histogram shows the distribution of `log1p_user_product_avg_cart_order` and helps identify skewness, concentration around common values and possible outliers.

### log1p_user_product_last_order_number

![Histogram of log1p_user_product_last_order_number](figures/hist_log1p_user_product_last_order_number.png)

Observation: The histogram shows the distribution of `log1p_user_product_last_order_number` and helps identify skewness, concentration around common values and possible outliers.

### log1p_user_total_orders

![Histogram of log1p_user_total_orders](figures/hist_log1p_user_total_orders.png)

Observation: The histogram shows the distribution of `log1p_user_total_orders` and helps identify skewness, concentration around common values and possible outliers.

### log1p_user_avg_basket_size

![Histogram of log1p_user_avg_basket_size](figures/hist_log1p_user_avg_basket_size.png)

Observation: The histogram shows the distribution of `log1p_user_avg_basket_size` and helps identify skewness, concentration around common values and possible outliers.

### log1p_user_max_basket_size

![Histogram of log1p_user_max_basket_size](figures/hist_log1p_user_max_basket_size.png)

Observation: The histogram shows the distribution of `log1p_user_max_basket_size` and helps identify skewness, concentration around common values and possible outliers.

### log1p_user_total_products

![Histogram of log1p_user_total_products](figures/hist_log1p_user_total_products.png)

Observation: The histogram shows the distribution of `log1p_user_total_products` and helps identify skewness, concentration around common values and possible outliers.

### log1p_user_total_reorders

![Histogram of log1p_user_total_reorders](figures/hist_log1p_user_total_reorders.png)

Observation: The histogram shows the distribution of `log1p_user_total_reorders` and helps identify skewness, concentration around common values and possible outliers.

### log1p_product_total_purchases

![Histogram of log1p_product_total_purchases](figures/hist_log1p_product_total_purchases.png)

Observation: The histogram shows the distribution of `log1p_product_total_purchases` and helps identify skewness, concentration around common values and possible outliers.

### log1p_product_total_reorders

![Histogram of log1p_product_total_reorders](figures/hist_log1p_product_total_reorders.png)

Observation: The histogram shows the distribution of `log1p_product_total_reorders` and helps identify skewness, concentration around common values and possible outliers.

### log1p_product_avg_add_to_cart_order

![Histogram of log1p_product_avg_add_to_cart_order](figures/hist_log1p_product_avg_add_to_cart_order.png)

Observation: The histogram shows the distribution of `log1p_product_avg_add_to_cart_order` and helps identify skewness, concentration around common values and possible outliers.

### log1p_product_first_cart_count

![Histogram of log1p_product_first_cart_count](figures/hist_log1p_product_first_cart_count.png)

Observation: The histogram shows the distribution of `log1p_product_first_cart_count` and helps identify skewness, concentration around common values and possible outliers.

### log1p_orders_since_last_purchase

![Histogram of log1p_orders_since_last_purchase](figures/hist_log1p_orders_since_last_purchase.png)

Observation: The histogram shows the distribution of `log1p_orders_since_last_purchase` and helps identify skewness, concentration around common values and possible outliers.



## 12. Boxplot Analysis

Boxplots were generated to identify outliers and spread in numerical features.


### user_product_total_orders

![Boxplot of user_product_total_orders](figures/box_user_product_total_orders.png)

Observation: The boxplot helps identify whether `user_product_total_orders` contains extreme values that may influence models, especially distance-based or linear models.

### user_product_total_reorders

![Boxplot of user_product_total_reorders](figures/box_user_product_total_reorders.png)

Observation: The boxplot helps identify whether `user_product_total_reorders` contains extreme values that may influence models, especially distance-based or linear models.

### user_product_avg_cart_order

![Boxplot of user_product_avg_cart_order](figures/box_user_product_avg_cart_order.png)

Observation: The boxplot helps identify whether `user_product_avg_cart_order` contains extreme values that may influence models, especially distance-based or linear models.

### user_product_last_order_number

![Boxplot of user_product_last_order_number](figures/box_user_product_last_order_number.png)

Observation: The boxplot helps identify whether `user_product_last_order_number` contains extreme values that may influence models, especially distance-based or linear models.

### user_total_orders

![Boxplot of user_total_orders](figures/box_user_total_orders.png)

Observation: The boxplot helps identify whether `user_total_orders` contains extreme values that may influence models, especially distance-based or linear models.

### user_avg_days_since_prior_order

![Boxplot of user_avg_days_since_prior_order](figures/box_user_avg_days_since_prior_order.png)

Observation: The boxplot helps identify whether `user_avg_days_since_prior_order` contains extreme values that may influence models, especially distance-based or linear models.

### user_avg_basket_size

![Boxplot of user_avg_basket_size](figures/box_user_avg_basket_size.png)

Observation: The boxplot helps identify whether `user_avg_basket_size` contains extreme values that may influence models, especially distance-based or linear models.

### user_max_basket_size

![Boxplot of user_max_basket_size](figures/box_user_max_basket_size.png)

Observation: The boxplot helps identify whether `user_max_basket_size` contains extreme values that may influence models, especially distance-based or linear models.

### user_total_products

![Boxplot of user_total_products](figures/box_user_total_products.png)

Observation: The boxplot helps identify whether `user_total_products` contains extreme values that may influence models, especially distance-based or linear models.

### user_total_reorders

![Boxplot of user_total_reorders](figures/box_user_total_reorders.png)

Observation: The boxplot helps identify whether `user_total_reorders` contains extreme values that may influence models, especially distance-based or linear models.

### user_reorder_rate

![Boxplot of user_reorder_rate](figures/box_user_reorder_rate.png)

Observation: The boxplot helps identify whether `user_reorder_rate` contains extreme values that may influence models, especially distance-based or linear models.

### product_total_purchases

![Boxplot of product_total_purchases](figures/box_product_total_purchases.png)

Observation: The boxplot helps identify whether `product_total_purchases` contains extreme values that may influence models, especially distance-based or linear models.

### product_total_reorders

![Boxplot of product_total_reorders](figures/box_product_total_reorders.png)

Observation: The boxplot helps identify whether `product_total_reorders` contains extreme values that may influence models, especially distance-based or linear models.

### product_avg_add_to_cart_order

![Boxplot of product_avg_add_to_cart_order](figures/box_product_avg_add_to_cart_order.png)

Observation: The boxplot helps identify whether `product_avg_add_to_cart_order` contains extreme values that may influence models, especially distance-based or linear models.

### product_reorder_rate

![Boxplot of product_reorder_rate](figures/box_product_reorder_rate.png)

Observation: The boxplot helps identify whether `product_reorder_rate` contains extreme values that may influence models, especially distance-based or linear models.

### product_first_cart_count

![Boxplot of product_first_cart_count](figures/box_product_first_cart_count.png)

Observation: The boxplot helps identify whether `product_first_cart_count` contains extreme values that may influence models, especially distance-based or linear models.

### product_first_cart_rate

![Boxplot of product_first_cart_rate](figures/box_product_first_cart_rate.png)

Observation: The boxplot helps identify whether `product_first_cart_rate` contains extreme values that may influence models, especially distance-based or linear models.

### user_product_reorder_rate

![Boxplot of user_product_reorder_rate](figures/box_user_product_reorder_rate.png)

Observation: The boxplot helps identify whether `user_product_reorder_rate` contains extreme values that may influence models, especially distance-based or linear models.

### orders_since_last_purchase

![Boxplot of orders_since_last_purchase](figures/box_orders_since_last_purchase.png)

Observation: The boxplot helps identify whether `orders_since_last_purchase` contains extreme values that may influence models, especially distance-based or linear models.

### user_product_purchase_share

![Boxplot of user_product_purchase_share](figures/box_user_product_purchase_share.png)

Observation: The boxplot helps identify whether `user_product_purchase_share` contains extreme values that may influence models, especially distance-based or linear models.

### log1p_user_product_total_orders

![Boxplot of log1p_user_product_total_orders](figures/box_log1p_user_product_total_orders.png)

Observation: The boxplot helps identify whether `log1p_user_product_total_orders` contains extreme values that may influence models, especially distance-based or linear models.

### log1p_user_product_total_reorders

![Boxplot of log1p_user_product_total_reorders](figures/box_log1p_user_product_total_reorders.png)

Observation: The boxplot helps identify whether `log1p_user_product_total_reorders` contains extreme values that may influence models, especially distance-based or linear models.

### log1p_user_product_avg_cart_order

![Boxplot of log1p_user_product_avg_cart_order](figures/box_log1p_user_product_avg_cart_order.png)

Observation: The boxplot helps identify whether `log1p_user_product_avg_cart_order` contains extreme values that may influence models, especially distance-based or linear models.

### log1p_user_product_last_order_number

![Boxplot of log1p_user_product_last_order_number](figures/box_log1p_user_product_last_order_number.png)

Observation: The boxplot helps identify whether `log1p_user_product_last_order_number` contains extreme values that may influence models, especially distance-based or linear models.

### log1p_user_total_orders

![Boxplot of log1p_user_total_orders](figures/box_log1p_user_total_orders.png)

Observation: The boxplot helps identify whether `log1p_user_total_orders` contains extreme values that may influence models, especially distance-based or linear models.

### log1p_user_avg_basket_size

![Boxplot of log1p_user_avg_basket_size](figures/box_log1p_user_avg_basket_size.png)

Observation: The boxplot helps identify whether `log1p_user_avg_basket_size` contains extreme values that may influence models, especially distance-based or linear models.

### log1p_user_max_basket_size

![Boxplot of log1p_user_max_basket_size](figures/box_log1p_user_max_basket_size.png)

Observation: The boxplot helps identify whether `log1p_user_max_basket_size` contains extreme values that may influence models, especially distance-based or linear models.

### log1p_user_total_products

![Boxplot of log1p_user_total_products](figures/box_log1p_user_total_products.png)

Observation: The boxplot helps identify whether `log1p_user_total_products` contains extreme values that may influence models, especially distance-based or linear models.

### log1p_user_total_reorders

![Boxplot of log1p_user_total_reorders](figures/box_log1p_user_total_reorders.png)

Observation: The boxplot helps identify whether `log1p_user_total_reorders` contains extreme values that may influence models, especially distance-based or linear models.

### log1p_product_total_purchases

![Boxplot of log1p_product_total_purchases](figures/box_log1p_product_total_purchases.png)

Observation: The boxplot helps identify whether `log1p_product_total_purchases` contains extreme values that may influence models, especially distance-based or linear models.

### log1p_product_total_reorders

![Boxplot of log1p_product_total_reorders](figures/box_log1p_product_total_reorders.png)

Observation: The boxplot helps identify whether `log1p_product_total_reorders` contains extreme values that may influence models, especially distance-based or linear models.

### log1p_product_avg_add_to_cart_order

![Boxplot of log1p_product_avg_add_to_cart_order](figures/box_log1p_product_avg_add_to_cart_order.png)

Observation: The boxplot helps identify whether `log1p_product_avg_add_to_cart_order` contains extreme values that may influence models, especially distance-based or linear models.

### log1p_product_first_cart_count

![Boxplot of log1p_product_first_cart_count](figures/box_log1p_product_first_cart_count.png)

Observation: The boxplot helps identify whether `log1p_product_first_cart_count` contains extreme values that may influence models, especially distance-based or linear models.

### log1p_orders_since_last_purchase

![Boxplot of log1p_orders_since_last_purchase](figures/box_log1p_orders_since_last_purchase.png)

Observation: The boxplot helps identify whether `log1p_orders_since_last_purchase` contains extreme values that may influence models, especially distance-based or linear models.



## 13. Scatter Plot Analysis

Scatter plots were generated for selected feature pairs using a sample of the dataset to keep the plots readable.


### user_product_total_orders vs user_product_reorder_rate

![Scatter plot](figures/scatter_user_product_total_orders_vs_user_product_reorder_rate.png)

Observation: This plot helps assess whether `user_product_total_orders` and `user_product_reorder_rate` show a visible relationship, clustering pattern or non-linear behaviour.

### product_total_purchases vs product_reorder_rate

![Scatter plot](figures/scatter_product_total_purchases_vs_product_reorder_rate.png)

Observation: This plot helps assess whether `product_total_purchases` and `product_reorder_rate` show a visible relationship, clustering pattern or non-linear behaviour.

### orders_since_last_purchase vs user_product_reorder_rate

![Scatter plot](figures/scatter_orders_since_last_purchase_vs_user_product_reorder_rate.png)

Observation: This plot helps assess whether `orders_since_last_purchase` and `user_product_reorder_rate` show a visible relationship, clustering pattern or non-linear behaviour.

### user_total_orders vs user_product_total_orders

![Scatter plot](figures/scatter_user_total_orders_vs_user_product_total_orders.png)

Observation: This plot helps assess whether `user_total_orders` and `user_product_total_orders` show a visible relationship, clustering pattern or non-linear behaviour.

### product_total_reorders vs product_reorder_rate

![Scatter plot](figures/scatter_product_total_reorders_vs_product_reorder_rate.png)

Observation: This plot helps assess whether `product_total_reorders` and `product_reorder_rate` show a visible relationship, clustering pattern or non-linear behaviour.



## 14. Correlation and Relationship Analysis

The correlation matrix below shows pairwise linear relationships between numerical features and the target.

![Correlation Matrix](figures/correlation_matrix.png)

### Correlation with Target

|                                      |   reordered_next_order |
|:-------------------------------------|-----------------------:|
| user_product_purchase_share          |             0.356217   |
| log1p_user_product_total_orders      |             0.277085   |
| log1p_user_product_total_reorders    |             0.274736   |
| user_product_total_reorders          |             0.248867   |
| user_product_total_orders            |             0.248867   |
| user_product_reorder_rate            |             0.248704   |
| product_reorder_rate                 |             0.163534   |
| log1p_product_total_reorders         |             0.137009   |
| log1p_product_first_cart_count       |             0.135729   |
| product_total_purchases              |             0.124987   |
| product_total_reorders               |             0.124832   |
| log1p_product_total_purchases        |             0.121915   |
| product_first_cart_count             |             0.113167   |
| user_avg_days_since_prior_order      |             0.0694485  |
| product_first_cart_rate              |             0.0550302  |
| user_avg_basket_size                 |             0.0495319  |
| log1p_user_avg_basket_size           |             0.0390558  |
| log1p_user_product_last_order_number |             0.0374582  |
| user_product_last_order_number       |             0.019598   |
| user_reorder_rate                    |             0.00988841 |
| user_max_basket_size                 |            -0.00817719 |
| log1p_user_max_basket_size           |            -0.0173107  |
| log1p_user_product_avg_cart_order    |            -0.0353541  |
| user_total_reorders                  |            -0.0393001  |
| user_product_avg_cart_order          |            -0.0435343  |
| log1p_user_total_reorders            |            -0.0488948  |
| user_total_products                  |            -0.0552416  |
| log1p_user_total_products            |            -0.0744598  |
| product_avg_add_to_cart_order        |            -0.078005   |
| log1p_product_avg_add_to_cart_order  |            -0.07838    |
| user_total_orders                    |            -0.0909542  |
| log1p_user_total_orders              |            -0.103873   |
| orders_since_last_purchase           |            -0.176472   |
| log1p_orders_since_last_purchase     |            -0.280195   |

### Observation

Features with higher positive correlation to the target may be useful predictors of next-basket reorder behaviour.

Correlation only captures linear relationships. Tree-based models such as Random Forest, XGBoost, LightGBM and CatBoost can capture non-linear interactions.



## 15. Feature vs Target Analysis

The mean value of each feature was compared between target classes.

|                                      |           0 |           1 |
|:-------------------------------------|------------:|------------:|
| user_product_total_orders            |   2.15926   |    5.26536  |
| user_product_total_reorders          |   1.15926   |    4.26536  |
| user_product_avg_cart_order          |   9.27358   |    8.25547  |
| user_product_last_order_number       |  16.2016    |   17.3905   |
| user_total_orders                    |  26.8649    |   19.7643   |
| user_avg_days_since_prior_order      |  12.3747    |   13.9167   |
| user_avg_basket_size                 |  12.5173    |   13.5666   |
| user_max_basket_size                 |  24.1665    |   23.8553   |
| user_total_products                  | 324.864     |  263.862    |
| user_total_reorders                  | 206.829     |  171.773    |
| user_reorder_rate                    |   0.502569  |    0.509135 |
| product_total_purchases              | 499.779     | 1090.44     |
| product_total_reorders               | 348.2       |  826.404    |
| product_avg_add_to_cart_order        |   8.87698   |    8.17432  |
| product_reorder_rate                 |   0.49202   |    0.612016 |
| product_first_cart_count             |  56.7312    |  159.35     |
| product_first_cart_rate              |   0.0880975 |    0.104164 |
| user_product_reorder_rate            |   0.237799  |    0.523004 |
| orders_since_last_purchase           |  10.6633    |    2.37381  |
| user_product_purchase_share          |   0.132878  |    0.328834 |
| log1p_user_product_total_orders      |   0.982374  |    1.48452  |
| log1p_user_product_total_reorders    |   0.435381  |    1.12301  |
| log1p_user_product_avg_cart_order    |   2.1126    |    2.03261  |
| log1p_user_product_last_order_number |   2.3793    |    2.50174  |
| log1p_user_total_orders              |   2.98574   |    2.68832  |
| log1p_user_avg_basket_size           |   2.50464   |    2.5647   |
| log1p_user_max_basket_size           |   3.12535   |    3.09819  |
| log1p_user_total_products            |   5.32617   |    5.07185  |
| log1p_user_total_reorders            |   4.5522    |    4.3205   |
| log1p_product_total_purchases        |   4.59705   |    5.38636  |
| log1p_product_total_reorders         |   3.88328   |    4.88228  |
| log1p_product_avg_add_to_cart_order  |   2.25493   |    2.18311  |
| log1p_product_first_cart_count       |   2.32554   |    3.1275   |
| log1p_orders_since_last_purchase     |   1.83964   |    0.744649 |

### Observation

This comparison shows how customer-product behaviour differs between products that were reordered and products that were not reordered.

Products that were reordered are expected to have stronger customer-product history, higher reorder rates and stronger recency signals.



## 16. Additional Useful Analysis

### Top Products by Frequency

| product_name           |   count |
|:-----------------------|--------:|
| Banana                 |    1811 |
| Bag of Organic Bananas |    1549 |
| Organic Strawberries   |    1458 |
| Organic Baby Spinach   |    1314 |
| Large Lemon            |    1142 |
| Limes                  |    1083 |
| Strawberries           |    1075 |
| Organic Avocado        |    1033 |
| Organic Hass Avocado   |    1015 |
| Organic Yellow Onion   |     871 |
| Organic Blueberries    |     869 |
| Organic Garlic         |     834 |
| Organic Zucchini       |     813 |
| Cucumber Kirby         |     775 |
| Organic Raspberries    |     753 |
| Organic Grape Tomatoes |     735 |
| Yellow Onions          |     724 |
| Organic Lemon          |     675 |
| Seedless Red Grapes    |     674 |
| Extra Virgin Olive Oil |     664 |

### Business Observations

- Frequently purchased products can dominate recommendations, so the model should balance popularity with personalisation.
- Recency and reorder history are important because grocery shopping contains repeated habits.
- Product-level reorder rate helps identify staple products.
- User-product interaction features are more personalised than global product popularity features.
- The class imbalance suggests that PR-AUC and Precision@K are important metrics for evaluating recommendation quality.


