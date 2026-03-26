import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_dataset(df):
    # 1. Διαγραφή γραμμών χωρίς target (RainTomorrow)
    df = df.dropna(subset=['RainTomorrow']).copy()
    # 2. Μετατροπή Target σε numeric (0/1) - Απαραίτητο για PCA και Μοντέλα
    df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1})
    # 3. Handle Date: Εξαγωγή Μήνα (Feature Engineering 1)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df = df.drop(columns=['Date'])
    return df


def feature_engineering(df):
    df = df.copy()
    if "MaxTemp" in df.columns and "MinTemp" in df.columns:
        df["TempRange"] = df["MaxTemp"] - df["MinTemp"]
    if "Humidity3pm" in df.columns and "Humidity9am" in df.columns:
        df["HumidityDiff"] = df["Humidity3pm"] - df["Humidity9am"]
    if "Pressure3pm" in df.columns and "Pressure9am" in df.columns:
        df["PressureDiff"] = df["Pressure3pm"] - df["Pressure9am"]
    if "RainToday" in df.columns:
        df["RainToday"] = df["RainToday"].map({'No': 0, 'Yes': 1})
    return df


def split_data(df):
    X = df.drop(columns=["RainTomorrow"])
    y = df["RainTomorrow"]
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.111, stratify=y_temp, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


def handle_missing(X_train, X_val, X_test):
    X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()
    for col in X_train.columns:
        val = X_train[col].median() if pd.api.types.is_numeric_dtype(X_train[col]) else X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(val)
        X_val[col] = X_val[col].fillna(val)
        X_test[col] = X_test[col].fillna(val)
    return X_train, X_val, X_test


def handle_outliers_and_encode(X_train, X_val, X_test):
    X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        Q1, Q3 = X_train[col].quantile(0.25), X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        X_train[col] = X_train[col].clip(lower, upper)
        X_val[col] = X_val[col].clip(lower, upper)
        X_test[col] = X_test[col].clip(lower, upper)

    X_train = pd.get_dummies(X_train)
    X_val = pd.get_dummies(X_val).reindex(columns=X_train.columns, fill_value=0)
    X_test = pd.get_dummies(X_test).reindex(columns=X_train.columns, fill_value=0)
    return X_train, X_val, X_test


def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, "models/scaler.pkl")
    return X_train_s, X_val_s, X_test_s


def run_pca_analysis(X_train_scaled, y_train):
    pca = PCA()
    pca_data = pca.fit_transform(X_train_scaled)

    # 1. Scree Plot
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.title('Cumulative Explained Variance (Scree Plot)')
    plt.xlabel('Number of Components')
    plt.ylabel('Variance Explained')
    plt.grid()
    plt.savefig('pca_scree_plot.png')

    # 2. 2D Scatter Plot
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=y_train, alpha=0.3)
    plt.title('PCA 2D Projection of Weather Data')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig('pca_2d_projection.png')

    print(f"PCA plots saved. Top 2 components explain {sum(pca.explained_variance_ratio_[:2]) * 100:.2f}% of variance.")
    return pca