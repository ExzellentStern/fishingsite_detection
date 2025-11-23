# =========================
# ロジスティック回帰を使用したフィッシングサイトの検出
# =========================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score
)
from sklearn.model_selection import (
    train_test_split, cross_val_score, cross_validate, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import optuna
import numpy as np

# --- 1) データ読み込み ---
df = pd.read_csv("dataset.csv")

# 最後の列がラベル想定（違うなら列名で y = df['label'].values に変える）
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# --- 2) 学習/テスト分割（層化）---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 3) ベースライン評価（スケーリング込みの公平な比較）---
baseline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)),
])

# 3-1) 交差検証（学習データ内）
baseline_cv_scores = cross_val_score(baseline, X_train, y_train, cv=5)
print("Baseline CV accuracy (k=5): {:.2f}%".format(100 * baseline_cv_scores.mean()))

# 3-2) テスト評価
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)
baseline_acc = accuracy_score(y_test, baseline_pred) * 100
print("Baseline test accuracy: {:.2f}%".format(baseline_acc))

# --- 4) Optuna でハイパラ探索（StratifiedKFold + Pipeline）---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

class Objective:
    def __init__(self, X, y, cv):
        self.X = X
        self.y = y
        self.cv = cv

    def __call__(self, trial):
        params = {
            "solver": trial.suggest_categorical(
                "solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
            ),
            "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
            "max_iter": trial.suggest_int("max_iter", 200, 5000),
        }
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**params, random_state=42)),
        ])

        scores = cross_validate(
            model, X=self.X, y=self.y, scoring="accuracy",
            cv=self.cv, n_jobs=-1, return_train_score=False
        )
        return scores["test_score"].mean()

objective = Objective(X_train, y_train, cv)
study = optuna.create_study(direction="maximize")
study.optimize(objective, timeout=60)

print("Best params:", study.best_params)
print("Best CV accuracy: {:.2f}%".format(100 * study.best_value))

# --- 5) ベストパラメータで再学習してテスト評価 ---
best_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        solver=study.best_params["solver"],
        C=study.best_params["C"],
        max_iter=study.best_params["max_iter"],
        random_state=42
    )),
])

best_model.fit(X_train, y_train)
pred = best_model.predict(X_test)

print("Tuned test accuracy: {:.2f}%".format(100 * accuracy_score(y_test, pred)))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

# 多クラスでも安全に集計（バランスよく見るなら macro）
print("Precision (macro): {:.2f}%".format(100 * precision_score(y_test, pred, average="macro")))
print("Recall (macro): {:.2f}%".format(100 * recall_score(y_test, pred, average="macro")))
