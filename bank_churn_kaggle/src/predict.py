import pandas as pd
from src.config import *

def predict_test(model, X_test, test_raw):
    pred = model.predict(X_test, num_iteration=model.best_iteration)
    sub = test_raw[["id"]].copy()
    sub[TARGET] = pred
    sub.to_csv(SUBMIT_CSV, index=False)
    print(f"预测结果已保存至 {SUBMIT_CSV}")
    return sub