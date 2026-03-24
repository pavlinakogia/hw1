import pandas as pd
import joblib
import os
from src.preprocessing import *
from src.train_classical import *
from src.train_neural import *
from src.evaluate import *

if not os.path.exists("models"): os.makedirs("models")

# 1. Load & Preprocess
df = pd.read_csv("data/dataset.csv")
df = prepare_dataset(df)
df = feature_engineering(df)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
X_train, X_val, X_test = handle_missing(X_train, X_val, X_test)
X_train, X_val, X_test = handle_outliers_and_encode(X_train, X_val, X_test)
X_train_s, X_val_s, X_test_s = scale_data(X_train, X_val, X_test)

# 2. PCA
run_pca(X_train_s, y_train)

# 3. Training
rf_model = train_random_forest(X_train_s, y_train)
lr_model = train_logistic_regression(X_train_s, y_train)
nn_model = train_neural_network(X_train_s, y_train, X_val_s, y_val)

# 4. Evaluation & Best Model Choice
acc_rf = evaluate_model(rf_model, X_test_s, y_test, "Random Forest")
acc_nn = evaluate_model(nn_model, X_test_s, y_test, "Neural Network")

best_model = rf_model if acc_rf >= acc_nn else nn_model
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(rf_model, "models/classical_model.pkl")

print(f"\nBest model saved: {'Random Forest' if acc_rf >= acc_nn else 'Neural Network'}")