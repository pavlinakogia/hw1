# Hands-on AI - Homework 1: Rain Prediction in Australia

## 1. Domain & Dataset
This project predicts whether it will rain tomorrow in Australia. The dataset contains ~10,000 weather observations.

## 2. Preprocessing & Leakage Prevention
- **Missing Values:** Imputed with median (numeric) and mode (categorical).
- **Outliers:** Handled via IQR clipping.
- **Scaling:** StandardScaler was used.
- **Leakage:** All statistics were calculated ONLY on the training set and then applied to validation/test sets.

## 3. Feature Engineering
- `PressureDiff`: Difference between 3pm and 9am pressure. (Intuition: Dropping pressure is a strong indicator of rain).
- `TempRange`: MaxTemp - MinTemp.
- `Month`: Extracted from Date to capture seasonality.

## 4. PCA Insights
- **Scree Plot:** Showed that about 15 components capture ~85% of the variance.
- **2D Projection:** Revealed that while classes overlap, there is a distinct clustering for "Rain" days.

## 5. Model Comparison (Test Set Results)
| Model | Accuracy | F1-Score (Rain) |
|-------|----------|-----------------|
| Random Forest | 83% | 0.59 |
| **Neural Network** | **84%** | **0.65** |

**Conclusion:** The Neural Network performed better, especially in capturing the "Rain" class (higher F1), likely due to its ability to model non-linear interactions better than the RF in this specific dataset size.

## 6. Best Model Designation
The **Neural Network** is saved as `best_model.pkl`. It is chosen for its superior precision-recall balance and higher overall accuracy.

## 7. Installation & Execution
1. Clone the repo: `git clone <your-repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the pipeline: `python main.py`