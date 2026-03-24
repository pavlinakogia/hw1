from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def train_random_forest(X_train, y_train):
    print("Training RF with GridSearchCV...")
    rf = RandomForestClassifier(random_state=42, class_weight="balanced")
    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def train_logistic_regression(X_train, y_train):
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    return lr