import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier  # 引入 XGBoost 分類器


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

# 8. 模型訓練與預測（XGBoost）
xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# 9. 評估指標
print("=== Logistic Regression ===")
print(classification_report(y_test, y_pred_logreg))

print("=== Random Forest ===")
print(classification_report(y_test, y_pred_rf))

print("=== XGBoost ===")
print(classification_report(y_test, y_pred_xgb))

# 計算評估指標 - Logistic Regression
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
precision_logreg = precision_score(y_test, y_pred_logreg)
recall_logreg = recall_score(y_test, y_pred_logreg)
f1_logreg = f1_score(y_test, y_pred_logreg)

print("Logistic Regression 評估指標")
print(f"Accuracy: {accuracy_logreg}")
print(f"Precision: {precision_logreg}")
print(f"Recall: {recall_logreg}")
print(f"F1-Score: {f1_logreg}")

# 計算評估指標 - Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print("Random Forest 評估指標")
print(f"Accuracy: {accuracy_rf}")
print(f"Precision: {precision_rf}")
print(f"Recall: {recall_rf}")
print(f"F1-Score: {f1_rf}")

# 計算評估指標 - XGBoost
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)

print("XGBoost 評估指標")
print(f"Accuracy: {accuracy_xgb}")
print(f"Precision: {precision_xgb}")
print(f"Recall: {recall_xgb}")
print(f"F1-Score: {f1_xgb}")

# 10. 檢查 XGBoost 是否過擬合
# 使用交叉驗證評估模型穩定性
cv_scores = cross_val_score(xgb, X_train, y_train, cv=5, scoring='accuracy')
print(f"XGBoost 5折交叉驗證準確率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 11. 特徵重要性比較
plt.figure(figsize=(12, 10))

# Random Forest 特徵重要性
plt.subplot(2, 1, 1)
feat_importances_rf = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances_rf.sort_values().plot(kind='barh', title='Random Forest Feature Importance')

# XGBoost 特徵重要性
plt.subplot(2, 1, 2)
feat_importances_xgb = pd.Series(xgb.feature_importances_, index=X.columns)
feat_importances_xgb.sort_values().plot(kind='barh', title='XGBoost Feature Importance')

plt.tight_layout()
plt.show()

# 12. 模型性能比較
models = ['Logistic Regression', 'Random Forest', 'XGBoost']
accuracy_scores = [accuracy_logreg, accuracy_rf, accuracy_xgb]
precision_scores = [precision_logreg, precision_rf, precision_xgb]
recall_scores = [recall_logreg, recall_rf, recall_xgb]
f1_scores = [f1_logreg, f1_rf, f1_xgb]

plt.figure(figsize=(15, 10))

# 準確率比較
plt.subplot(2, 2, 1)
plt.bar(models, accuracy_scores, color=['blue', 'green', 'red'])
plt.title('Accuracy comparison')
plt.ylim([min(accuracy_scores) * 0.95, 1.0])

# 精確率比較
plt.subplot(2, 2, 2)
plt.bar(models, precision_scores, color=['blue', 'green', 'red'])
plt.title('Comparison of accuracy') # 精確率比較
plt.ylim([min(precision_scores) * 0.95, 1.0])

# 召回率比較
plt.subplot(2, 2, 3)
plt.bar(models, recall_scores, color=['blue', 'green', 'red'])
plt.title('Recall comparison') # 召回率比較
plt.ylim([min(recall_scores) * 0.95, 1.0])

# F1分數比較
plt.subplot(2, 2, 4)
plt.bar(models, f1_scores, color=['blue', 'green', 'red'])
plt.title('F1 Score Comparison') # F1 分數比較
plt.ylim([min(f1_scores) * 0.95, 1.0])

plt.tight_layout()
plt.show()

# 13. 混淆矩陣比較
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
cm_logreg = confusion_matrix(y_test, y_pred_logreg)
sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Real Label')

plt.subplot(1, 3, 2)
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Real Label')

plt.subplot(1, 3, 3)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Reds', cbar=False)
plt.title('XGBoost Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Real Label')

plt.tight_layout()
plt.show()