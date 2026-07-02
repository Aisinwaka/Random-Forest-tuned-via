# Random Forest Classification — Airline Customer Satisfaction Analysis

## Dataset Overview

**Source:** `Invistico_Airline.csv`
**Size:** 129,880 passenger survey responses, 22 columns (129,487 after dropping 393 rows with missing `Arrival Delay in Minutes`)
**Target:** `satisfaction` (satisfied = 1, dissatisfied = 0) — roughly balanced at 55% / 45%
**Features:** passenger demographics (Age, Customer Type, Type of Travel, Class), flight details (Distance, Departure/Arrival Delay), and 14 service-quality ratings (seat comfort, wifi, entertainment, boarding, baggage handling, etc.)

Categorical columns (`Customer Type`, `Type of Travel`, `Class`) were one-hot encoded before modeling.

## Split Strategy

A **three-way split** (60% train / 20% validation / 20% test) was used instead of a simple train/test split. The validation set was fixed and passed to `GridSearchCV` via `PredefinedSplit`, so every hyperparameter combination was scored on exactly the same held-out rows — and the test set was never touched until final evaluation. This avoids the leakage risk of using the test set (or overlapping CV folds) during model selection.

## Hyperparameter Tuning Approach

`GridSearchCV` searched over:
- `max_depth`: [10, 20, 30, None]
- `n_estimators`: [50, 100, 150]
- `min_samples_leaf`: [1, 2, 5]

scored by F1 on the predefined validation fold (36 candidate combinations, 1 fold each since the split is fixed).

**Best parameters found:** `max_depth=30`, `n_estimators=100`, `min_samples_leaf=1`
**Best validation F1-score:** 0.9557

## Results: Random Forest vs. Decision Tree (Test Set)

| Metric | Decision Tree | Random Forest |
|---|---|---|
| Accuracy | 0.9296 | **0.9518** |
| Precision | 0.9347 | **0.9644** |
| Recall | 0.9369 | **0.9468** |
| F1-score | 0.9358 | **0.9555** |

The tuned Random Forest outperforms the single Decision Tree on every metric. Just as important, its **train–test accuracy gap is smaller** than the Decision Tree's, confirming the ensemble generalizes better and overfits less to training data.

## Top Satisfaction Drivers (Feature Importance)

1. Inflight entertainment (0.219)
2. Seat comfort (0.137)
3. Ease of online booking (0.074)
4. Online support (0.066)
5. Leg room service (0.047)
6. Food and drink (0.044)
7. Flight distance (0.040)
8. On-board service (0.040)
9. Customer loyalty status (0.036)
10. Online boarding (0.033)

## Business Recommendation

Inflight entertainment and seat comfort together account for over a third of the model's predictive signal — these are the highest-leverage areas for the airline to invest in if the goal is moving the needle on customer satisfaction. Digital touchpoints (online booking, online support, online boarding) form a secondary cluster worth bundling into a "digital experience" improvement track. Because the Random Forest is more stable and less prone to overfitting than a single Decision Tree, it is the recommended model for any production deployment (e.g., flagging at-risk passengers for service recovery in real time).

## Files

- `random_forest_airline.ipynb` — full notebook, all cells executed, with GridSearchCV output, metrics, comparison, and feature importance plots
- `README.md` — this file
