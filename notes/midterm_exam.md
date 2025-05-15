機器學習與 AI、深度學習的關係為何？
A. 深度學習是 AI 的一種，機器學習則不屬於 AI
B. 機器學習包含 AI 與深度學習
C. AI 包含機器學習，機器學習又包含深度學習
D. AI 是機器學習的子集
正解：C

若企業希望從財務報表與銷售數據中擷取商業洞察，應使用下列哪種技術？
A. 深度學習（DL）
B. 感測器資料分析
C. 機器學習（ML）
D. 強化學習（RL）
正解：C

哪一個 Python 套件最適合用來處理結構化表格數據？
A. NumPy
B. Matplotlib
C. Pandas
D. scikit-learn
正解：C

在 Pandas 中處理表格資料時，常使用哪種資料結構？
A. List
B. DataFrame
C. Array
D. Dictionary
正解：B

下列哪一項屬於 scikit-learn 的功能？
A. 網頁爬蟲
B. 互動式圖表
C. 分類與迴歸演算法
D. 多媒體處理
正解：C

若要查看 DataFrame 前五筆資料，應使用哪個 Pandas 函數？
A. df.info()
B. df.describe()
C. df.head(5)
D. df.first(5)
正解：C

Pandas 中哪個方法可以用來填補缺失值？
A. df.fillna()
B. df.remove_na()
C. df.drop_null()
D. df.complete()
正解：A

在 Pandas 中使用哪個方法可以根據欄位值進行彙總？
A. df.total()
B. df.groupby()
C. df.sort()
D. df.aggregate()
正解：B

下列程式碼會產生什麼樣的輸出？
np.random.seed(42)
print(np.random.rand(2))
A. 每次執行都會有不同的結果
B. [0.37454012 0.95071431]
C. 產生整數
D. 程式會出錯
正解：B

以下哪一行程式會正確印出 “Name” 欄的資料？
df = pd.DataFrame({'Name': ['Amy', 'Ben', 'Carl'], 'Age': [22, 35, 27]})
A. df[Name]
B. df['Name']
C. df.get("Name")
D. df.name
正解：B

下列哪個方法可以用 0 取代 DataFrame 中的 NaN 值？
A. df.replace(0)
B. df.fillna(0)
C. df.dropna()
D. df.fillmissing(0)
正解：B

對下列資料使用 df.groupby('Category').sum()，會對哪一欄進行加總？
data = {'Category': ['A', 'B', 'A', 'B'], 'Value': [100, 200, 150, 250]}
df = pd.DataFrame(data)
A. Category
B. Value
C. Index
D. 無欄位被加總
正解：B

要選取欄位 'Name'，哪個語法正確？
A. df['Name']
B. df.loc[:, 'Name']
C. df.iloc[:, 0]（若 'Name' 是第 0 欄）
D. 以上皆可
正解：D

df.iloc[1:4, 1] 會取出什麼？
A. 第 1、2、3 列的第 1 欄
B. 第 1 到第 4 列的第 1 欄
C. 第 1 列到第 4 列的所有欄位
D. 第 1 到 4 欄的第 1 列
正解：A

在 EDA 中，以下哪一種圖表最適合用來檢查數值型資料的異常值？
A. 長條圖（Bar Chart）
B. 散點圖（Scatter Plot）
C. 箱型圖（Box Plot）
D. 折線圖（Line Chart）
正解：C

使用以下哪個函數可以將類別型欄位轉換為 One-Hot Encoding？
A. LabelEncoder()
B. pd.get_dummies()
C. MinMaxScaler()
D. StandardScaler()
正解：B

特徵交互（Feature Interaction）的目的為何？
A. 將類別欄位轉成數值欄位
B. 合併不同特徵以創造新特徵，增強模型能力
C. 將資料轉換為常態分布
D. 減少資料筆數以避免過擬合
正解：B

在進行資料標準化時，以下哪個類別最常用於將資料轉換為平均值為 0、標準差為 1？
A. MinMaxScaler()
B. StandardScaler()
C. Normalizer()
D. Binarizer()
正解：B

下列哪一項最符合監督式學習（Supervised Learning）的特徵？
A. 模型透過與環境互動學習最佳策略
B. 模型使用帶有標籤的資料來學習輸入與輸出間的關係
C. 模型不需要標籤，自動學習資料結構
D. 模型結合標籤資料與未標籤資料進行訓練
正解：B

下列哪一個是回歸模型中常見的誤差評估指標？
A. Accuracy
B. Precision
C. Mean Squared Error (MSE)
D. F1 Score
正解：C

在 Python 的線性回歸模型中，要取得模型的斜率（係數），應使用哪個屬性？
A. model.slope_
B. model.coef_
C. model.beta_
D. model.gradient_
正解：B

若模型預測結果的 "R平方" 值等於 1，這代表什麼？
A. 模型與資料毫無關聯
B. 模型效果不如隨機猜測
C. 模型能完美解釋所有變異
D. 模型出現過擬合
正解：C

若使用 train_test_split(X, y, test_size=0.2)，代表資料集的哪一部分用於測試集？
A. 20%
B. 80%
C. 50%
D. 42%
正解：A

執行 model.predict(X_test) 的目的是？
A. 建立模型
B. 評估準確率
C. 儲存模型
D. 產出對新資料的預測結果
正解：D

關於邏輯回歸（Logistic Regression）的敘述，下列何者正確？
A. 是用來解決回歸問題的演算法
B. 輸出值是一個離散的整數
C. 是用來解決二元分類問題的演算法
D. 一定需要標準化後才能使用
正解：C

邏輯回歸中，以下哪一個函數負責將對數勝算（log-odds）轉換為機率？
A. ReLU
B. tanh
C. Sigmoid
D. Softmax
正解：C

當我們希望模型不要漏掉任何正類樣本時，應提升哪一個指標？
A. Accuracy
B. Precision
C. Recall
D. F1 Score
正解：C

當特徵分裂後的加權 Entropy 越小，代表？
A. 該特徵無意義
B. 分裂後的純度越高，資訊增益越大
C. 分裂後的混亂越大
D. 模型容易過擬合
正解：B

決策樹演算法在選擇分裂點時，常用的純度衡量指標為何？
A. MSE
B. RMSE
C. ROC 曲線
D. 基尼不純度（Gini Impurity）
正解：D

在 DecisionTreeClassifier 中，哪個參數可以設定樹的最大深度？
A. depth_limit
B. max_level
C. tree_depth
D. max_depth
正解：D

在分類模型中，ROC 曲線的 X 軸與 Y 軸分別為？
A. Recall / Accuracy
B. Precision / Recall
C. False Positive Rate / True Positive Rate
D. True Negative Rate / Precision
正解：C

在分類任務中，隨機森林如何決定最終預測結果？
A. 計算平均機率
B. 使用最大機率法則
C. 多數決（Majority Voting）
D. 使用最小特徵數原則
正解：C

若要取得分類模型的混淆矩陣，應使用哪一個函式？
A. confusion_matrix()
B. classification_matrix()
C. matrix_score()
D. confusion_score()
正解：A


若在 K-means 中群中心仍持續變動，演算法會怎麼做？
A. 停止訓練
B. 回傳初始值
C. 重複分群與更新中心直到收斂
D. 重新指定群數 K
正解：C

輪廓係數（Silhouette Score）越接近哪個數值表示分群品質越好？
A. -1
B. 0
C. 0.5
D. 1
正解：D

下列哪一個分群演算法能自動處理離群值（outliers）？
A. KMeans()
B. AgglomerativeClustering()
C. DBSCAN()
D. MiniBatchKMeans()
正解：C