import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


def run_pca(X_train, y_train):
    pca = PCA(random_state=42)
    X_pca = pca.fit_transform(X_train)

    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.title("Scree Plot")
    plt.savefig("models/pca_scree.png")

    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, alpha=0.3)
    plt.title("PCA 2D Projection")
    plt.savefig("models/pca_2d.png")
    plt.show()


def evaluate_model(model, X, y, name):
    preds = model.predict(X)
    print(f"\n--- {name} ---")
    print(classification_report(y, preds))
    return (preds == y).mean()


def plot_all(model, X, y, name):
    preds = model.predict(X)
    cm = confusion_matrix(y, preds)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"CM: {name}")
    plt.savefig(f"models/cm_{name}.png")
    plt.show()