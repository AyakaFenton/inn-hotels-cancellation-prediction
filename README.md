# inn-hotels-cancellation-prediction
# INN Hotels — Booking Cancellation Prediction

**Tools:** Python · scikit-learn · statsmodels · pandas · matplotlib · seaborn  
**Type:** Supervised Learning / Binary Classification

---

## Overview

INN Hotels faces significant revenue loss due to booking cancellations. This project builds machine learning models to predict which bookings are likely to be canceled, enabling the hotel to take proactive measures such as overbooking strategies, reconfirmation requests, or deposit policies.

**Dataset:** 36,275 bookings × 18 features (lead time, room type, market segment, special requests, price per room, etc.)  
**Target:** `booking_status` — Canceled (32.9%) vs. Not Canceled (67.1%)

---

## Key Results

### Logistic Regression (Test Set)

*Threshold tuning was applied to optimize F1-score for cancellation detection.*

| Model | Accuracy | Recall | Precision | F1 |
|---|---|---|---|---|
| LR — Default Threshold (0.5) | 0.805 | 0.631 | 0.729 | 0.676 |
| LR — Threshold 0.42 | 0.803 | 0.704 | 0.694 | 0.699 |
| **LR — Threshold 0.37** | **0.796** | **0.740** | **0.666** | **0.701** |

✅ **Best Logistic Regression:** Threshold 0.37 — highest F1 (0.701), no overfitting observed

---

### Decision Tree (Test Set)

*GridSearchCV for pre-pruning; cost-complexity pruning (CCP) for post-pruning.*

| Model | Accuracy | Recall | Precision | F1 |
|---|---|---|---|---|
| DT — Default | 0.872 | 0.817 | 0.794 | 0.806 |
| DT — Pre-Pruning | 0.835 | 0.783 | 0.728 | 0.754 |
| **DT — Post-Pruning** | **0.869** | **0.854** | **0.768** | **0.809** |

✅ **Final Model: Post-Pruned Decision Tree** — F1: 0.809 (Train: 0.849), best balance of performance and generalization

> The post-pruned decision tree substantially outperforms logistic regression (F1: 0.701 vs. 0.809), while pruning effectively reduced overfitting compared to the default tree.

---

## Key Findings

**Top predictors of cancellation:**
- `lead_time` — Longer booking lead times strongly increase cancellation likelihood
- `avg_price_per_room` — Lower room prices associated with higher cancellation rates
- `no_of_special_requests` — More special requests significantly reduce cancellation probability
- `market_segment_type_Online` — Online bookings cancel more frequently than offline/corporate

**High-risk cancellation profile (from decision tree rules):**
- Lead time > 151.5 days
- Average price per room ≤ $100
- No special requests (≤ 0.5)
- Number of adults ≤ 1.5
- Arrival month before December

---

## Business Recommendations

1. **Flag high-risk bookings proactively** — Apply the model at booking time to identify likely cancellations; trigger reconfirmation emails or deposit requests for high-risk profiles.
2. **Manage lead time risk** — Bookings made more than 151 days in advance warrant closer monitoring; consider incentives (early-bird discounts) to reduce cancellation intent.
3. **Leverage special requests as a retention signal** — Guests with special requests cancel far less; actively encourage guests to submit preferences at booking.
4. **Review online channel pricing** — Online bookings with low room prices show the highest cancellation rates; dynamic pricing or stricter cancellation policies for low-rate online bookings may reduce revenue loss.
5. **Retrain periodically** — Seasonal and behavioral patterns shift; model should be updated with fresh data regularly.

---

## ML Techniques Demonstrated

- Exploratory Data Analysis (univariate, bivariate, target-stratified)
- Outlier treatment (IQR method)
- Feature encoding and multicollinearity check (p-value based feature selection)
- Logistic Regression with threshold tuning (AUC-ROC optimal threshold)
- Decision Tree with GridSearchCV (pre-pruning) and cost-complexity pruning (post-pruning)
- Evaluation: Accuracy, Recall, Precision, F1-score, Confusion Matrix
- Model comparison and final model selection

---

## Files

| File | Description |
|---|---|
| `INN_Hotels_Analysis.ipynb` | Full notebook with code, outputs, and commentary |
