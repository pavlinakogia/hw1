import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def train_classical_model(X_train, y_train):
    print("Starting Classical Model Training (Random Forest)...")

    rf = RandomForestClassifier(random_state=42, class_weight="balanced")

    # Αυτό καλύπτει το Task 5 (Bonus - Hyperparameter Tuning)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }

    grid = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"Best Parameters: {grid.best_params_}")

    # Αποθήκευση (Task 3)
    joblib.dump(best_model, "models/classical_model.pkl")
    print("Classical model saved to models/classical_model.pkl")

    return best_model