import pandas as pd
import joblib
import os
import shutil
from src.preprocessing import (
    prepare_dataset, feature_engineering, split_data,
    handle_missing, handle_outliers_and_encode, scale_data
)
from src.train_classical import train_classical_model
from src.train_neural import train_neural_network
from src.evaluate import run_pca_analysis, evaluate_and_compare, designate_best_model


def main():
    # Δημιουργία φακέλων
    if not os.path.exists("models"):
        os.makedirs("models")

    print("--- 1. Data Loading & Preprocessing ---")

    try:
        df = pd.read_csv("data/dataset.csv")
    except FileNotFoundError:
        print("Error: dataset.csv not found in data/ folder. Loading from current directory...")
        df = pd.read_csv("dataset.csv")

    # Pipeline Προεπεξεργασίας
    df = prepare_dataset(df)
    df = feature_engineering(df)

    # Split (80/10/10)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Imputation, Encoding & Scaling (Derived from Train only)
    X_train, X_val, X_test = handle_missing(X_train, X_val, X_test)
    X_train, X_val, X_test = handle_outliers_and_encode(X_train, X_val, X_test)
    X_train_s, X_val_s, X_test_s = scale_data(X_train, X_val, X_test)

    joblib.dump(X_train.columns.tolist(), "models/feature_names.pkl")

    print("--- 2. PCA Analysis (Task 2 & 5) ---")
    # Εκτέλεση PCA και αποθήκευση γραφημάτων
    run_pca_analysis(X_test_s, y_test)

    print("--- 3. Training Models (Task 3) ---")
    # Εκπαίδευση Classical (με GridSearchCV / Tuning)
    rf_model = train_classical_model(X_train_s, y_train)

    # Εκπαίδευση Neural Network (με Early Stopping & NNWrapper)
    nn_wrapper = train_neural_network(X_train_s, y_train, X_val_s, y_val)

    print("--- 4. Evaluation & Model Comparison (Task 4 & 6) ---")

    models_to_compare = {
        "Classical": rf_model,
        "Neural": nn_wrapper
    }

    # Side-by-side σύγκριση, metrics και γραφήματα
    results_df = evaluate_and_compare(models_to_compare, X_test_s, y_test)

    print("--- 5. Designating Best Model (Task 4) ---")

    best_model_name = designate_best_model(results_df, models_to_compare)

    print(f"\n Pipeline completed successfully!")
    print(f" Best Model: {best_model_name}")
    print(f" Scaler saved at models/scaler.pkl")
    print(f" Figures saved in models/ folder")


if __name__ == "__main__":
    main()