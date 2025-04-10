import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# 1. 讀取資料
df = pd.read_csv("datasets/video_games_sales.csv")
df = df.dropna()

# 2. 建立新目標欄位：是否在 NA 超過百萬銷量
df['Is_NA_Hit'] = df['NA_Sales'] > 1

# 3. 處理類別變數（Label Encoding）
categorical_cols = ['Platform', 'Genre', 'Publisher']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 4. 選擇特徵與標籤
X = df[['Platform', 'Genre', 'Publisher', 'EU_Sales', 'JP_Sales', 'Global_Sales']]
y = df['Is_NA_Hit']

# 5. 拆分資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 模型訓練與預測（Logistic Regression）
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

# 7. 模型訓練與預測（Random Forest）
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 8. 評估指標
print("=== Logistic Regression ===")
print(classification_report(y_test, y_pred_logreg))

print("=== Random Forest ===")
print(classification_report(y_test, y_pred_rf))

# 計算評估指標
accuracy = accuracy_score(y_test, y_pred_logreg)
precision = precision_score(y_test, y_pred_logreg)
recall = recall_score(y_test, y_pred_logreg)
f1 = f1_score(y_test, y_pred_logreg)

print("Y Logistic Regression")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

accuracy = accuracy_score(y_test, y_pred_rf)
precision = precision_score(y_test, y_pred_rf)
recall = recall_score(y_test, y_pred_rf)
f1 = f1_score(y_test, y_pred_rf)

print("Y Random Forest")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# 9. 特徵重要性 (隨機森林)
feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances.sort_values().plot(kind='barh', title='Feature Importances')
plt.tight_layout()
plt.show()

