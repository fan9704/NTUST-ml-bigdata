from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
import numpy as np

# 建立 FastAPI App
app = FastAPI()

# 載入模型
model = joblib.load("xgb_ecg_model.joblib")

# 定義輸入資料格式
class ECGData(BaseModel):
    features: list[float]  # 預期是 187 維的 ECG 特徵

# 建立 API 路由
@app.post("/predict/")
def predict(data: ECGData):
    x = np.array(data.features).reshape(1, -1)
    pred = model.predict(x)
    return {"prediction": int(pred[0])}

# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    uvicorn.run(host="0.0.0.0",port=8000, app="app:app", reload=True)