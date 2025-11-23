# ===== Enron spam/ham → TF-IDF → LightGBM (Optuna TunerCV) =====
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import optuna.integration.lightgbm as olgb
import matplotlib.pyplot as plt

# 1) メール読み込み
def read_texts(dir_path: str):
    texts = []
    for name in os.listdir(dir_path):
        p = os.path.join(dir_path, name)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())
    return texts

spam_dir = "./enron1/spam"
ham_dir  = "./enron1/ham"

spam = read_texts(spam_dir)
ham  = read_texts(ham_dir)

# 2) DataFrame化（label: spam=1, ham=0）
all_mails = [(t, 1) for t in spam] + [(t, 0) for t in ham]
df = pd.DataFrame(all_mails, columns=["text", "label"]).sample(frac=1, random_state=42).reset_index(drop=True)

# 3) ベクトル化
tfidf = TfidfVectorizer(
    stop_words="english",
    lowercase=True,
    min_df=2,
    max_df=0.9,
    ngram_range=(1, 2)
)
X = tfidf.fit_transform(df["text"])
y = df["label"].to_numpy(dtype=int)

# 4) データ分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=101, stratify=y
)

# 5) LightGBM Dataset
train = olgb.Dataset(X_train, label=y_train)

# 6) パラメータ
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "boosting_type": "gbdt",
}

# 7) Optuna LightGBMTunerCV
tuner = olgb.LightGBMTunerCV(
    params,
    train,
    num_boost_round=1000
)
tuner.run()

best_params = tuner.best_params
best_iter   = tuner.best_iteration
print("Best params:", best_params)
print("Best iteration:", best_iter)

# 8) 最良パラメータで再学習
gbm = olgb.train(
    best_params,
    train,
    num_boost_round=best_iter,
    verbose_eval=False,
)

# 9) 予測・評価
probs = gbm.predict(X_test)
pred_labels = (probs >= 0.5).astype(int)
print("Accuracy: {:.5f} %".format(100 * accuracy_score(y_test, pred_labels)))
print("Confusion matrix:\n", confusion_matrix(y_test, pred_labels))

# ===== 10) 重要度プロット =====
import lightgbm as lgb
lgb.plot_importance(gbm, figsize=(12, 6), max_num_features=10)
plt.show()

# ===== 11) "subject" 含有回数の集計 =====
spam_rows = (df.label == 1)
spam_data = df[spam_rows]

count = 0
for i in spam_data['text']:
    count += i.count('subject')
print("Spam内の 'subject' 出現回数:", count)

legit_rows = (df.label == 0)
legit_data = df[legit_rows]

count = 0
for i in legit_data['text']:
    count += i.count('subject')
print("Ham内の 'subject' 出現回数:", count)
