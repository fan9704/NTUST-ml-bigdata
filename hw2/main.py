import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error,mean_absolute_error

# 讀取 CSV 檔案
df = pd.read_csv("datasets/student_performance_dataset.csv")

# 檢視數據基本資訊
print(df.head())
print(df.info())
print(df.describe())
# 處理遺漏值（如果有）
df = df.dropna()

# 選擇數值型變數進行標準化
num_cols = ['Study_Hours_per_Week', 'Attendance_Rate', 'Past_Exam_Scores', 'Final_Exam_Score']
scaler = StandardScaler() # 將數據轉換為 平均值為 0，標準差為 1 的標準常態分佈
df[num_cols] = scaler.fit_transform(df[num_cols]) # 會同時 計算平均值和標準差，並用這些值來轉換數據

# 簡單線性回歸（使用 Past_Exam_Scores 預測 Final_Exam_Score）
X_simple = df[['Past_Exam_Scores']]
y_simple = df['Final_Exam_Score']
X_train, X_test, y_train, y_test = train_test_split(X_simple, y_simple, test_size=0.2, random_state=42)

model_simple = LinearRegression()
model_simple.fit(X_train, y_train)
y_pred_simple = model_simple.predict(X_test)

mse_simple = mean_squared_error(y_test, y_pred_simple)
r2_simple = r2_score(y_test, y_pred_simple)

print(f"簡單線性回歸 MSE: {mse_simple}")
print(f"簡單線性回歸 R2: {r2_simple}")

# 多變量線性回歸（使用學習時數、出勤率、過去考試成績預測最終考試成績）
X_multi = df[['Study_Hours_per_Week', 'Attendance_Rate', 'Past_Exam_Scores']]
y_multi = df['Final_Exam_Score']
X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

model_multi = LinearRegression()
model_multi.fit(X_train, y_train)
y_pred_multi = model_multi.predict(X_test)

# 強調大誤差對模型的影響
mse_multi = mean_squared_error(y_test, y_pred_multi) # 均方誤差 - 越小代表模型越好
# 直觀解釋誤差大小，單位與數據一致 避免異常值影響，讓誤差更穩定
mae_multi = mean_absolute_error(y_test, y_pred_multi) # 平均絕對誤差 - 越小代表模型越好
# 強調大誤差對模型的影響 直觀解釋誤差大小，單位與數據一致
rmse_multi = root_mean_squared_error(y_test, y_pred_multi) # 均方根誤差 - 越小代表模型越好
# 判斷模型是高估還是低估，以及高估或低估的程度
mpe_multi = np.mean((y_test - y_pred_multi) / y_test) # 平均絕對百分比誤差 - 越小代表模型越好
# 關心預測誤差的比例 (百分比)
mape_multi = np.mean(np.abs((y_test - y_pred_multi) / y_test)) # 平均絕對百分比誤差 - 越小代表模型越好
r2_multi = r2_score(y_test, y_pred_multi) # 決定係數 - 越接近 1 代表模型越好

n = X_test.shape[0]
k = X_test.shape[1]
adj_r2_multi = 1 - ((1 - r2_multi) * (n - 1) / (n - k - 1)) # 調整後的決定係數 - 越接近 1 代表模型越好

print(f"多變量線性回歸 MSE: {mse_multi}")
print(f"多變量線性回歸 R2: {r2_multi}")
print(f"多變量線性回歸 Adjusted R2: {adj_r2_multi}")
print(f"平均絕對誤差 MAE: {mae_multi}")
print(f"平均絕對誤差 RMSE: {rmse_multi}")
print(f"平均絕對誤差 MPE: {mpe_multi}")
print(f"平均絕對誤差 MAPE: {mape_multi}")

# 繪製散點圖 - 過去考試成績 vs. 最終考試成績
fig1 = px.scatter(df, x='Past_Exam_Scores', y='Final_Exam_Score', trendline="ols", title="過去考試成績 vs. 最終考試成績")
fig1.show()

# 繪製散點圖 - 學習時數、出勤率對最終考試成績的影響
fig2 = px.scatter_3d(df, x='Study_Hours_per_Week', y='Attendance_Rate', z='Final_Exam_Score', title="學習時數、出勤率對最終考試成績的影響")
fig2.show()

correlation_matrix = df[num_cols].corr() # 計算數值型變數的相關性矩陣

# 繪製 Heatmap
# 傳入相關性矩陣 在格子內顯示數值（相關係數） cmap='coolwarm' → 設定顏色 數值顯示 2 位小數  網格線寬度
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()