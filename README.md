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
  
