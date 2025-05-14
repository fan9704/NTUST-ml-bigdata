import numpy as np 
import pandas as pd 
import numpy as np 
import random
import requests
import os
import warnings
import time

from sklearn.model_selection import train_test_split
from multiprocessing import cpu_count

n_cpu = cpu_count()
plot_block = True
os.environ["LOKY_MAX_CPU_COUNT"] = "16"
warnings.filterwarnings("ignore", category=UserWarning)
folder_path = "datasets"
output_path = "train2_images"
# 建立 output_path 資料夾
os.makedirs(output_path, exist_ok=True)

# 讀入資料
abnormal = pd.read_csv(f"{folder_path}/ptbdb_abnormal.csv", header = None) 
normal = pd.read_csv(f"{folder_path}/ptbdb_normal.csv", header = None)

# 移除 label 欄位
abnormal = abnormal.drop([187], axis=1)
normal = normal.drop([187], axis=1)

# 標籤建立
y_abnormal = np.ones((abnormal.shape[0]))
y_abnormal = pd.DataFrame(y_abnormal)

y_normal = np.zeros((normal.shape[0]))
y_normal = pd.DataFrame(y_normal)
# 合併資料
X = pd.concat([abnormal, normal], sort=True)
y = pd.concat([y_abnormal, y_normal] ,sort=True)
# 資料切分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# 結果計數器
correct_count = 0
incorrect_count = 0
while True:

    # 隨機抽取一筆資料
    idx = random.randint(0, len(X) - 1)
    sample = X.iloc[idx].values.tolist()
    true_label = y.iloc[idx].values[0]
    print(sample)

    # 準備 JSON
    payload = {"features": sample}

    # 發送 POST 請求（請確認 FastAPI 有啟動，預設 http://127.0.0.1:8000）
    url = "http://127.0.0.1:8000/predict"
    response = requests.post(url, json=payload)
    # 印出回應
    print(f"送出第 {idx} 筆資料")
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())
    if (response.json()["prediction"] == 0 and 0 == true_label) or (response.json()["prediction"] == 1 and 0 != true_label):
        # print("預測正確")
        correct_count += 1
    else:
        incorrect_count += 1
        print("預測錯誤")
        print("目前預測結果")
        print("正確數量:", correct_count)
        print("錯誤數量:", incorrect_count)
        time.sleep(5)