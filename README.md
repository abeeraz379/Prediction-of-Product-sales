# 🛒 Prediction of Product Sales

## 📌 Project Overview
This project analyzes the **Big Mart Sales dataset** through a full data science pipeline:
data cleaning, exploratory data analysis (EDA), and predictive modeling.
The goal is to uncover patterns in product and outlet features that drive sales,
and build a model to accurately predict **Item Outlet Sales**.

> 📎 **Dataset:** [Big Mart Sales III — Analytics Vidhya](https://www.analyticsvidhya.com/datahack/contest/practice-problem-big-mart-sales-iii/)

---

## 🧹 Data Cleaning
- Checked for and removed duplicate records
- Handled missing values using appropriate placeholders
- Resolved data inconsistencies across categorical columns
- Saved the cleaned dataset for downstream analysis

---

## 📊 Exploratory Data Analysis (EDA)
A comprehensive univariate and bivariate EDA was performed on all numeric
and categorical columns using boxplots, heatmaps, histograms, and count plots.

### Count Plots — Categorical Feature Distributions
<img width="1634" height="1739" alt="countplot" src="https://github.com/user-attachments/assets/2c90e1fe-51e8-4cf4-98e7-45628c894bfd" />

- **Outlet Size:** Medium outlets are most common; High-sized outlets are the least frequent.
- **Outlet Type:** Supermarket Type1 dominates overwhelmingly over all other types.
- **Item Fat Content:** Low Fat items are approximately 2× more frequent than Regular items.
- **Item Type:** Fruits & Vegetables and Snack Foods are the most stocked; Seafood is the rarest.

---

### Heatmap — Feature Correlation Matrix
<img width="686" height="589" alt="heatmap" src="https://github.com/user-attachments/assets/3851894f-05a6-4b23-b491-9ec68da9655b" />

Most features show weak correlations with one another. The two strongest relationships are:
- **Item_Weight ↔ Outlet_Establishment_Year** (r = 0.54)
- **Item_MRP ↔ Item_Outlet_Sales** (r = 0.57) — suggesting that item price
is the most influential numeric predictor of sales.

---

### Boxplot — Outlet Establishment Year vs Item Outlet Sales
<img width="589" height="455" alt="boxplot" src="https://github.com/user-attachments/assets/cc259d0a-bf55-41cf-8432-5d1fdf0b7312" />

-  Sales distributions are relatively consistent across most outlet establishment years.
**1985** outlets show the highest variability, while **1998** outlets have
notably lower and more compressed sales figures.

  ---
  ## Preprocessing
  ---

A `ColumnTransformer` was built to handle three types of features simultaneously, ensuring all data was clean and properly formatted before being fed into any model.

**Numeric features** were first passed through a `SimpleImputer` using the mean strategy to fill in any missing values, then scaled using `StandardScaler` to standardize all numeric columns to the same scale, preventing any single feature from dominating the model due to its magnitude.

**Ordinal features** were processed through a `SimpleImputer` to handle missing values, followed by an `OrdinalEncoder` to convert ordered categorical values into meaningful integers that preserve their natural ranking, and finally a `StandardScaler` to bring them onto the same scale as the numeric features.

**Categorical features** were handled with a `SimpleImputer` to address any missing values, then transformed using `OneHotEncoder` to convert nominal categorical variables into binary columns, allowing the model to interpret them without assuming any order or ranking between categories.

This entire preprocessing pipeline was structured using scikit-learn's `ColumnTransformer`, which applies each set of transformations to the appropriate columns in parallel, ensuring a clean, consistent, and reproducible preprocessing workflow across both training and test data.

 
  ###  Recommended Model: Tuned Random Forest Regressor

After evaluating all three models, I recommend the **Tuned Random Forest Regressor** as the best model to move forward with.

---

### Justification

Looking at the results across all three models:

- **Linear Regression** had consistent training and test scores (R² ≈ 0.56), meaning it is neither overfitting nor underfitting — but it simply doesn't explain the data well enough, leaving nearly 44% of the variance unexplained.

- **Untuned Random Forest** performed exceptionally well on training data (R² = 0.939) but collapsed on test data (R² = 0.561), which is a clear sign of severe overfitting. It essentially memorized the training data and failed to generalize.

- **Tuned Random Forest** struck the best balance — it reduced the overfitting seen in the untuned version while achieving a better test R² (0.592) than both Linear Regression and the untuned Random Forest. The gap between training and test scores is much more reasonable, making it the most generalizable and reliable model of the three.

---
### Model Insights 
#### Linear regression
<img width="623" height="547" alt="top3" src="https://github.com/user-attachments/assets/4c49f56b-fab3-4a10-81dd-7016d53ef539" />

#### Linear regression without intercept
<img width="604" height="470" alt="lr" src="https://github.com/user-attachments/assets/268a478e-bfd2-4082-9c7e-776ea6ea8ca0" />

##### Results 
<img width="756" height="323" alt="image" src="https://github.com/user-attachments/assets/9dad2c59-2776-4e6d-b551-d18d9cff5900" />

#### random forest
<img width="583" height="547" alt="top5" src="https://github.com/user-attachments/assets/f490229d-ac8b-40d1-a40f-0229d7b4457e" />

##### Results 
<img width="852" height="340" alt="image" src="https://github.com/user-attachments/assets/b4b38f00-bf23-4eee-baf0-26b6116c179d" />

###  Model Performance for Non-Technical Stakeholders

**R-Squared (R²):**

Our recommended model — the Tuned Random Forest — has an R² of **0.592 on test data**. Think of R² as a score that tells us how well our model explains what drives the outcome we're predicting. A score of 0.592 means that our model can explain about **59% of the variation** in the target variable. In simple terms, if you imagine 100 different cases in our data, our model correctly accounts for the patterns behind roughly 59 of them. While there is still room for improvement, this is a meaningful result and gives us a solid foundation to build on.

---

**Selected Metric: MAE (Mean Absolute Error)**

I chose **MAE** to communicate the model's error to stakeholders because it is the most straightforward and intuitive metric — it simply tells us, on average, how far off our predictions are from the actual values, in the same units as what we're predicting.

Our Tuned Random Forest has a test MAE of **760.35**, meaning that on average, our model's predictions are off by about **760 units** from the real value. Unlike RMSE, MAE doesn't exaggerate the impact of large errors, which makes it easier to interpret and communicate honestly to a non-technical audience.

---

###  Overfitting / Underfitting Analysis

| Model | Train R² | Test R² | Gap | Assessment |
|---|---|---|---|---|
| Linear Regression | 0.564 | 0.562 | 0.002 | Underfit — too simple |
| Untuned Random Forest | 0.939 | 0.561 | 0.378 | Severely overfit |
| **Tuned Random Forest** | **0.703** | **0.592** | **0.111** | **Mild overfit — best balance** |

The Tuned Random Forest shows a training R² of 0.703 versus a test R² of 0.592, a gap of about **0.111**. This tells us the model is **slightly overfit** — it performs somewhat better on data it has seen than on new data — but this gap is significantly smaller than the untuned version's gap of 0.378. The tuning process (via GridSearchCV) successfully reined in the overfitting by constraining the model's complexity, making it much more trustworthy when applied to real-world, unseen data.
