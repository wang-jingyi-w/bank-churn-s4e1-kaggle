# run.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from category_encoders import TargetEncoder
import lightgbm as lgb

# ========== 1. 读取数据（路径改为本地 data/raw/） ==========
PATH = "data/raw/"
train = pd.read_csv(PATH + "train.csv")
test = pd.read_csv(PATH + "test.csv")
sub = pd.read_csv(PATH + "sample_submission.csv")
TARGET = "Exited"

print("训练集形状:", train.shape, "测试集形状:", test.shape)

# ========== 2. 划分训练集和验证集（分层抽样，保持图片中的逻辑） ==========
X_full = train.drop(["Exited", "id", "CustomerId", "Surname"], axis=1)
y_full = train["Exited"]
X_test_raw = test.drop(["id", "CustomerId", "Surname"], axis=1)

X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

# ========== 3. 目标编码（Target Encoding，避免数据泄露） ==========
te_geo = TargetEncoder()
te_gender = TargetEncoder()

# 训练集 fit + transform
X_train["Geography"] = te_geo.fit_transform(X_train["Geography"], y_train)
X_train["Gender"] = te_gender.fit_transform(X_train["Gender"], y_train)

# 验证集和测试集只 transform
X_val["Geography"] = te_geo.transform(X_val["Geography"])
X_val["Gender"] = te_gender.transform(X_val["Gender"])

X_test_raw["Geography"] = te_geo.transform(X_test_raw["Geography"])
X_test_raw["Gender"] = te_gender.transform(X_test_raw["Gender"])

# ========== 4. 特征工程（新增3个特征） ==========
def feature_eng(df):
    df = df.copy()
    df["ZeroBalance"] = (df["Balance"] == 0).astype(int)
    df["Age_Product"] = df["Age"] * df["NumOfProducts"]
    df["Balance_Salary"] = df["Balance"] / (df["EstimatedSalary"] + 1e-6)
    return df

X_train = feature_eng(X_train)
X_val = feature_eng(X_val)
X_test = feature_eng(X_test_raw)

# ========== 5. 训练 LightGBM 模型 ==========
params = {
    "objective": "binary",
    "metric": "auc",
    "random_state": 42,
    "verbosity": -1
}

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

model = lgb.train(
    params,
    lgb_train,
    num_boost_round=3000,
    valid_sets=[lgb_val],
    callbacks=[lgb.early_stopping(stopping_rounds=200)]
)

# ========== 6. 评估验证集 AUC ==========
val_pred = model.predict(X_val, num_iteration=model.best_iteration)
auc_score = roc_auc_score(y_val, val_pred)
print(f"\n🎯 验证集 AUC: {auc_score:.6f}")

# ========== 7. 预测测试集并生成提交文件 ==========
test_pred = model.predict(X_test, num_iteration=model.best_iteration)
sub["Exited"] = test_pred
sub.to_csv("submission.csv", index=False)
print("✅ 提交文件已生成: submission.csv")