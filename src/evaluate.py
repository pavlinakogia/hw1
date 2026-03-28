import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shutil
import os
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve,
    auc, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
)
from sklearn.decomposition import PCA


def run_pca_analysis(X_test_scaled, y_test):
    """
    Task 2 & 5: PCA Insights.
    """
    print("\n--- Running PCA Analysis ---")
    pca = PCA(random_state=42)
    X_pca = pca.fit_transform(X_test_scaled)

    # 1. Scree Plot
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
    plt.title("Scree Plot: Cumulative Explained Variance")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Variance")
    plt.grid()
    plt.savefig("models/pca_scree.png")

    # 2. 2D Projection
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_test, alpha=0.4, palette='viridis')
    plt.title("PCA 2D Projection")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig("models/pca_2d.png")
    plt.show()


def plot_evaluation_graphs(y_test, y_pred, y_probs, name):
    """
    Confusion Matrix και ROC Curve για κάθε μοντέλο.
    """
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix: {name}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"models/cm_{name.lower()}.png")
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"ROC Curve: {name}")
    plt.legend(loc="lower right")
    plt.savefig(f"models/roc_{name.lower()}.png")
    plt.close()


def evaluate_and_compare(models_dict, X_test, y_test):
    """
    Task 4 & 6: Σύγκριση μοντέλων side-by-side.
    """
    all_results = []

    for name, model in models_dict.items():
        print(f"\nEvaluating {name}...")

        # Προβλέψεις
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]  # Πιθανότητες για την κλάση "Yes" (1)

        # Υπολογισμός Metrics
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_probs)
        }
        all_results.append(metrics)

        # Εκτύπωση Classification Report στο console
        print(classification_report(y_test, y_pred))

        # Δημιουργία Γραφημάτων
        plot_evaluation_graphs(y_test, y_pred, y_probs, name)

    # Μετατροπή σε DataFrame για τη σύγκριση "Side-by-Side"
    df_comparison = pd.DataFrame(all_results)
    print("\n--- Final Model Comparison Table ---")
    print(df_comparison.to_string(index=False))

    return df_comparison


def designate_best_model(df_results, models_dict):
    """
    Task 4: Επιλογή του καλύτερου μοντέλου βάσει F1-Score και αποθήκευση.
    """
    # Επιλέγουμε τον νικητή βάσει F1-Score (πιο αξιόπιστο για imbalanced datasets)
    best_row = df_results.loc[df_results['F1-Score'].idxmax()]
    best_name = best_row['Model']
    best_model_obj = models_dict[best_name]

    print(f"\n The Best Model is: {best_name} with F1-Score: {best_row['F1-Score']:.4f}")

    # Αποθήκευση ως best_model.pkl
    joblib.dump(best_model_obj, "models/best_model.pkl")
    print(f"Successfully saved {best_name} as models/best_model.pkl")

    return best_name