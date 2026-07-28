import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import *

def load_raw_data():
    """读取原始数据集"""
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    
    # 缺失值统计
    missing = train.isnull().sum()
    missing = missing[missing > 0]
    print("训练集缺失值统计：\n", missing)
    return train, test

def split_dataset(train_df):
    """划分训练集、验证集，分层采样保证正负样本比例"""
    X_full = train_df.drop([TARGET] + DROP_COLS, axis=1)
    y_full = train_df[TARGET]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full,
        test_size=TEST_SPLIT_RATIO,
        random_state=RANDOM_SEED,
        stratify=y_full
    )
    print(f"Train:{X_train.shape}, Val:{X_val.shape}")
    return X_train, X_val, y_train, y_val