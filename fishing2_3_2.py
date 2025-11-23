from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import codecs
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# --- 先に: メール読み込みユーティリティ ---
def init_lists(folder):
    key_list = []
    file_list = os.listdir(folder)
    for filename in file_list:
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as f:
                key_list.append(f.read())
    return key_list

# --- メール読み込み → ラベル付け ---
spam = init_lists('./enron1/spam/')
ham  = init_lists('./enron1/ham/')
all_mails = [(mail, '1') for mail in spam]
all_mails += [(mail, '0') for mail in ham]

# --- DataFrame作成 ---
df = pd.DataFrame(all_mails, columns=['text', 'label'])

# --- TF-IDFベクトライザ ---
tfidf = TfidfVectorizer(stop_words="english", lowercase=False)

# テキストをベクトル化
X_tfidf = tfidf.fit_transform(df['text'])

# 正しい列名取得
column_names = tfidf.get_feature_names_out()

# DataFrame化（元の方針を維持して密行列に変換）
X = pd.DataFrame(X_tfidf.toarray(), columns=column_names).astype('float')

# ラベルをfloatに変換
y = df['label'].astype('float')
