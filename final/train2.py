import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from matplotlib import colors 
from matplotlib.ticker import PercentFormatter 
import numpy as np 
import matplotlib.pyplot as plt 

import os
import warnings
import joblib
from multiprocessing import cpu_count
n_cpu = cpu_count()
plot_block = True
os.environ["LOKY_MAX_CPU_COUNT"] = "16"
warnings.filterwarnings("ignore", category=UserWarning)
print(os.listdir("datasets"))
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

print(abnormal.head)

flatten_ab_y = (abnormal.values)
flatten_ab_y  = flatten_ab_y[:,5:70].flatten()

print(flatten_ab_y.shape)

ab_x=np.arange(0,65)
ab_x = np.tile(ab_x, abnormal.shape[0])

plt.hist2d(ab_x, flatten_ab_y, bins=(65, 100), cmap=plt.cm.jet)
plt.title("Abnormal Signal Histogram")
plt.colorbar()
plt.savefig(f"{output_path}/abnormal_hist2d.png")
plt.close()


# 異常折線圖
for i in [0, 50, 117, 1111, 100]:
    plt.plot((abnormal.values)[i][5:70])
    plt.title(f"Abnormal Sample #{i}")
    plt.savefig(f"{output_path}/abnormal_{i}.png")
    plt.close()

flatten_norm_y = normal.values
flatten_norm_y  = flatten_norm_y[:,5:70].flatten()

norm_x=np.arange(0,65)
norm_x = np.tile(norm_x, normal.shape[0])

# 正常 hist2d
plt.hist2d(norm_x, flatten_norm_y, bins=(65, 100), cmap=plt.cm.jet)
plt.title("Normal Signal Histogram")
plt.colorbar()
plt.savefig(f"{output_path}/normal_hist2d.png")
plt.close()

# 正常折線圖
for i in [0, 50, 117, 1111, 100]:
    plt.plot((normal.values)[i][5:70])
    plt.title(f"Normal Sample #{i}")
    plt.savefig(f"{output_path}/normal_{i}.png")
    plt.close()

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

print(abnormal.shape)
print(normal.shape)
# 缺失值檢查
print(np.any(X_train.isna().sum()))
print(np.any(X_test.isna().sum()))

seed=123
# 多模型比較
classifiers = [
    LogisticRegression(class_weight='balanced', random_state=seed),
    KNeighborsClassifier(3, n_jobs=n_cpu),
    SVC(gamma='auto', class_weight='balanced', random_state=seed),
    RandomForestClassifier(random_state=seed, n_estimators = 1000),
    MLPClassifier(alpha=1, max_iter=1000),
    XGBClassifier(learning_rate =0.1,n_estimators=1000, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8,
                  colsample_bytree=0.8, objective= 'binary:logistic',nthread=4, scale_pos_weight=1,seed=seed)
]

names = ["Logistic", "Nearest Neighbors", "RBF SVM", "Random Forest", "Neural Net", "XGB"]

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.utils.validation import column_or_1d

for name, clf in zip(names, classifiers):
    clf.fit(X_train, column_or_1d(y_train, warn=True))
    print(f"==={name}===")
    y_pred = clf.predict(X_test)
    print(f"Precision: {round(precision_score(y_test, y_pred ),3)}")
    print(f"Accuracy: {round(accuracy_score(y_test, y_pred ),3)}")
    print(f"Recall: {round(recall_score(y_test, y_pred) ,3)}")
    print(f"F1-Score: {round(f1_score(y_test, y_pred ),3)}")


# feature importance
from matplotlib import pyplot as plt

feature_imp = np.argsort(clf.feature_importances_)
print(np.flip(feature_imp))

# 特徵重要性
plt.figure(figsize=(20, 8))
plt.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
plt.title("Feature Importance (XGB)")
plt.savefig(f"{output_path}/feature_importance.png")
plt.close()

# k-fold 交叉驗證
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, y.values.ravel(), cv=5, scoring='accuracy')
print("Cross-validation accuracy:", scores.mean())

# 輸出模型
joblib.dump(classifiers[-1], "xgb_ecg_model.joblib")