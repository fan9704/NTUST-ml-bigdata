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

# 1. 讀取資料
df = pd.read_csv("datasets/video_games_sales.csv")
df = df.dropna()

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 使用數值欄位做分群
features = df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Global_Sales']]

# Elbow Method
inertias = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features)
    inertias.append(kmeans.inertia_)

# Silhouette Score（找最佳 k 值）
for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(features)
    score = silhouette_score(features, labels)
    print(f"Silhouette Score for {k} clusters: {score:.3f}")

plt.plot(range(2, 10), inertias, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

