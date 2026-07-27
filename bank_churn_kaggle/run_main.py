from src.data_loader import load_raw_data, split_dataset
from src.eda import run_all_eda
from src.feature_engineering import create_new_features, target_encode
from src.train import train_lgb_model
from src.predict import predict_test

if __name__ == "__main__":
    # 1.加载数据
    train_raw, test_raw = load_raw_data()
    # 2.EDA可视化
    run_all_eda(train_raw)
    # 3.划分数据集
    X_train, X_val, y_train, y_val = split_dataset(train_raw)
    X_test = test_raw.drop(DROP_COLS, axis=1).copy()
    # 4.特征工程
    X_train = create_new_features(X_train)
    X_val = create_new_features(X_val)
    X_test = create_new_features(X_test)
    X_train, X_val, X_test = target_encode(X_train, X_val, X_test, y_train)
    # 5.训练模型
    model = train_lgb_model(X_train, X_val, y_train, y_val)
    # 6.测试集预测输出提交文件
    predict_test(model, X_test, test_raw)