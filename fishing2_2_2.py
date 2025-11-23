# =========================
# 決定木でフィッシング検出: Baseline→Optuna→評価
# =========================
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
import numpy as np
import pandas as pd
import optuna

# --- データ読み込み ---
df = pd.read_csv("dataset.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# --- 学習/テスト分割（層化）---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- CVスキーム ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Optuna目的関数 ---
class ObjectiveDTC:
    def __init__(self, X, y, cv):
        self.X = X
        self.y = y
        self.cv = cv

    def __call__(self, trial):
        params = {
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),  # sklearn新しめなら "log_loss" も可
            "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
            "max_depth": trial.suggest_int("max_depth", 2, 64),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 64),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 32),
            "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
            "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 0.02),  # cost-complexity pruning
        }

        model = DecisionTreeClassifier(random_state=42, **params)

        scores = cross_validate(
            model, X=self.X, y=self.y,
            scoring="accuracy",
            cv=self.cv,
            n_jobs=-1,
            return_train_score=False
        )
        return scores["test_score"].mean()

# --- 最適化 ---
objective = ObjectiveDTC(X_train, y_train, cv)
study = optuna.create_study(direction="maximize")
study.optimize(objective, timeout=60)

print("Best params:", study.best_params)
print("Best CV accuracy: {:.2f}%".format(100 * study.best_value))

# --- ベストで学習 & テスト評価 ---
best = study.best_params
best_model = DecisionTreeClassifier(random_state=42, **best)
best_model.fit(X_train, y_train)
pred = best_model.predict(X_test)

print("Test accuracy: {:.2f}%".format(100 * accuracy_score(y_test, pred)))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print("Precision (macro): {:.2f}%".format(100 * precision_score(y_test, pred, average="macro")))
print("Recall (macro): {:.2f}%".format(100 * recall_score(y_test, pred, average="macro")))

# --- 特徴量重要度（上位10） ---
importances = best_model.feature_importances_
idx = np.argsort(importances)[::-1][:10]
print("\nTop-10 feature importances:")
for rank, i in enumerate(idx, 1):
    print(f"{rank:2d}. feature[{i}] = {importances[i]:.4f}")
