# Quiz

---

## W5

### 1. 以下哪一種描述最符合監督式學習 (Supervised Learning) 的特徵？

- A 模型不需要標籤 (Label)，透過資料本身的結構來學習模式
- B 模型透過與環境互動，根據獎勵或懲罰來學習最佳策略
- C 模型使用帶有標籤的數據進行訓練，學習輸入與輸出之間的對應關係 [X]
- D 模型只使用少量的標籤數據，並結合大量的未標籤數據來學習

### 2. 線性回歸 (Linear Regression) 屬於以下哪一種類型的機器學習與應用問題？

- A 監督式學習 (Supervised Learning) - 分類問題 (Classification)
- B 監督式學習 (Supervised Learning) - 迴歸問題 (Regression) [X]
- C 非監督式學習 (Unsupervised Learning) - 聚類 (Clustering)
- D 強化學習 (Reinforcement Learning) - 決策問題 (Decision Making)

### 3. 當我們在 Python 中執行 train_test_split(X, y, test_size=0.2, random_state=42)，這表示我們將多少比例的數據用於測試集？

- A 20% [X]
- B 42%
- C 80%
- D 50%

### 4. 在 Python 中，若想要查看線性回歸模型的斜率 (係數)，應該使用哪個屬性？

- A model.slope_
- B model.coef_ [X]
- C model.weight_
- D model.beta_

### 5. 在多變量線性回歸中，當我們加入一個新的變數時，調整後決定係數 (Adjusted 𝑅2 ) 會如何變化？

- A 一定會上升，因為新增變數讓模型更複雜
- B 一定會下降，因為多餘的變數會降低模型準確度
- C 只有當新變數對模型有幫助時才會上升，否則可能會下降 [X]
- D 不會變化，因為它與普通 𝑅2 相同

---

## W6

### 1. 下列關於邏輯回歸（Logistic Regression）的敘述，何者正確？

- A 是用來解決回歸問題的演算法
- B 輸出值是一個離散的整數
- C 是一種用來解決二元分類的演算法 [X]
- D 必須先將資料標準化後才能使用

### 2. 當我們關心模型不漏掉任何正類樣本時，應該提升哪個指標？

- A Precision
- B Recall [X]
- C Accuracy
- D F1 Score

### 3. 若某特徵分裂後的加權 Entropy 越小，代表？

- A 該特徵無意義
- B 模型的準確率會下降
- C 分類的純度越低
- D 該特徵提供較高資訊增益 [X]

### 4. 隨機森林在分類任務中如何決定最終預測？

- A 計算平均機率
- B 多數決（Majority Voting）[X]
- C 最小損失原則
- D 最大特徵重要性
