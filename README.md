# KKBox Churn Prediction

Predicting subscriber churn for [KKBox](https://www.kkbox.com/), an Asian music
streaming service, using gradient-boosted trees. Built on the WSDM KKBox Churn
Prediction Challenge dataset.

A user is labeled **churned** if they do not renew their membership within 30
days of it expiring. The goal is not just to predict churn accurately, but to
predict it *early* using signals available before a user actually leaves, so a
retention team still has time to act.

## Overview

The project moves from exploratory analysis through feature engineering to a
final XGBoost model that predicts churn from leading behavioral signals. The
headline result: the model recalls **89% of churners at 0.971 validation AUC**
using only features that are available before a customer leaves.

## Data

The dataset comes from the Kaggle WSDM - KKBox's Churn Prediction Challenge:

**https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data**

It spans four tables, joined on the user ID (`msno`):

| Table | Contents |
|-------|----------|
| `train_v2.csv` | User IDs and the binary churn label |
| `members_v3.csv` | Demographics — age, gender, registration channel |
| `transactions_v2.csv` | Payment and plan history |
| `user_logs_v2.csv` | Daily listening activity |


## Approach

**Exploratory analysis.** The target is imbalanced (~9% churn), so accuracy is
not a meaningful metric — predicting "nobody churns" would score 91%. The
analysis instead leans on class balance, AUC, and per-class precision/recall.
EDA surfaced auto-renew status and listening engagement as the signals that most
clearly separate churners from stayers.

**Feature engineering.** A deliberately small set of features is built per
table rather than an exhaustive sweep:

- *Members* — account age, cleaned age, gender, registration channel
- *Transactions* — auto-renew rate, mean amount paid, mean discount, transaction count, payment recency
- *User logs* — average listening seconds, average unique songs, active days, total plays, listening recency

Missing values (from users with no transaction or listening history) are left in
place and handled natively by XGBoost, which learns a default split direction for
missing data. "No activity at all" is itself a meaningful disengagement signal,
so it is not overwritten by imputation.

**Modeling.** An XGBoost classifier with `scale_pos_weight` set to the
negative/positive ratio to counter class imbalance, plus early stopping on
validation AUC. Evaluation covers AUC-ROC, a per-class classification report, a
confusion matrix, and gain-based feature importance.

## Results

The final model predicts churn from **leading behavioral signals only** —
auto-renew status, engagement, and payment recency — reaching **0.971 validation
AUC-ROC** and recalling **89% of churners**.

| Metric | Stay | Churn |
|--------|------|-------|
| Precision | 0.99 | 0.56 |
| Recall | 0.93 | 0.89 |
| F1 | 0.96 | 0.68 |

At the default 0.5 threshold the model catches 15,514 of 17,466 actual churners.
Precision on the churn class is lower (0.56) for a retention use case this is
usually the right tradeoff, since a missed churner is more costly than an
unnecessary retention offer.

Auto-renew status dominates feature importance, followed by payment recency and
listening engagement.

### A note on feature leakage

An earlier version of this model included the feature "cancellation history" and scored 0.986
AUC. On inspection, that feature was leaking the outcome: churn is *defined* as
non-renewal within 30 days of expiry, and a cancellation is often the very act
that produces that non-renewal, so the feature was close to a restatement of
the label rather than a genuine predictor.

The final model excludes cancellation features for this reason. Removing the leak
cost only ~0.015 AUC (0.986 → 0.971), which shows that nearly all the predictive
signal comes from legitimate leading indicators that are available *before* a
user leaves and are therefore actually actionable for retention.
