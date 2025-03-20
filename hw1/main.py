import numpy as np
import pandas as pd
import plotly.express as px
# 讀取資料集
df = pd.read_csv("./hw1/datasets/pokemon_data_pokeapi.csv")
# 取前幾筆資料
print(df.head())
# 顯示資料集資訊
print(df.info())
# 描述資料集
print(df.describe())
# 處理遺漏值（如果有）
df = df.fillna(0)
# 轉換數據：計算 BMI（體重 / 身高^2）
df['BMI'] = df['Weight (kg)'] / (df['Height (m)'] ** 2)
# 聚合數據：計算不同類型寶可夢的平均 BMI
df_type_group = df.groupby('Type1')['BMI'].mean().reset_index()
# 繪製長條圖 - 不同類型寶可夢的平均 BMI
fig1 = px.bar(df_type_group, x='Type1', y='BMI', title="不同類型寶可夢的平均 BMI")
fig1.show()
# 繪製散點圖 - 身高 vs. 體重
fig2 = px.scatter(df, x='Height (m)', y='Weight (kg)', color='Type1', 
                  title="身高 vs. 體重")
fig2.show()
# 繪製長條圖 - 每個世代的傳說寶可夢數量
df_legendary_group = df.groupby('Generation')['Legendary Status'].apply(lambda x: (x == 'Yes').sum()).reset_index(name='Legendary Count')
fig3 = px.bar(df_legendary_group, x='Generation', y='Legendary Count', title="每個世代的傳說寶可夢數量")
fig3.show()