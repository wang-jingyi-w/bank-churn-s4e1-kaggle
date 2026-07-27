from category_encoders import TargetEncoder
from src.config import *

def create_new_features(df):
    """衍生特征，对应你截图的feature_eng函数"""
    df = df.copy()
    df["ZeroBalance"] = (df["Balance"] == 0).astype(int)
    df["Age_Product"] = df["Age"] * df["NumOfProducts"]
    df["Balance_Salary"] = df["Balance"] / (df["EstimatedSalary"] + 1e-6)
    return df

def target_encode(X_train, X_val, X_test, y_train):
    """目标编码，严格防止数据泄露：仅用训练集fit"""
    te_geo = TargetEncoder()
    te_gender = TargetEncoder()
    
    X_train["Geography"] = te_geo.fit_transform(X_train["Geography"], y_train)
    X_train["Gender"] = te_gender.fit_transform(X_train["Gender"], y_train)
    
    X_val["Geography"] = te_geo.transform(X_val["Geography"])
    X_val["Gender"] = te_gender.transform(X_val["Gender"])
    
    X_test["Geography"] = te_geo.transform(X_test["Geography"])
    X_test["Gender"] = te_gender.transform(X_test["Gender"])
    return X_train, X_val, X_test