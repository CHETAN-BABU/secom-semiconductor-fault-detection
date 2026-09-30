# SECOM Semiconductor Fault Detection

End-to-end data mining pipeline to predict pass/fail outcomes in semiconductor manufacturing from 591 sensor signals, built for a heavily imbalanced, high-dimensional, messy real-world dataset.

**Tools:** Python · pandas · scikit-learn · imbalanced-learn · matplotlib · seaborn

## Problem
The [UCI SECOM dataset](https://archive.ics.uci.edu/dataset/179/secom) has **1,567 production runs × 591 sensor features**, with many missing readings and a severe class imbalance (**93.4% pass / 6.6% fail**). The goal is to detect failing units early from sensor data.

## Approach
1. **Preprocessing:** removed constant or mostly-missing features, imputed missing values, standardised
2. **EDA:** class balance, high-variance sensor distributions by class, inter-sensor correlation
3. **Outlier analysis:** extreme readings kept on purpose, since they may be signs of a fault
4. **PCA:** reduced the feature space to the components that explain 95% of the variance
5. **K-Means clustering:** tested whether pass and fail units separate without labels
6. **Classification:** k-NN, Decision Tree and SVM, with **SMOTE** oversampling for the minority class
7. **Evaluation:** precision, recall, F1 and confusion matrices, focused on recall for the failure class

## Key findings
- PCA kept 95% of the variance with only a small fraction of the original features.
- K-Means (k=2) **could not** recover the pass/fail structure, so unsupervised methods alone aren't enough here.
- **Decision Tree + SMOTE** gave the best precision/recall balance and meaningful recall on the failure class.

## Repository
| Path | Contents |
|---|---|
| `secom_fault_detection.ipynb` | Full analysis notebook |
| `data/` | SECOM data and labels (UCI) |
| `report/report.pdf` | Full written report |

## Run it
```bash
pip install -r requirements.txt
jupyter notebook secom_fault_detection.ipynb
```

## Next steps
Cross-validated hyperparameter tuning, ensemble models (Random Forest, Gradient Boosting) and interpretable feature selection to pinpoint the faulty sensor channels.

---
*MSc Data Science & Analytics, Munster Technological University (Data Mining & Statistical Modelling)*
