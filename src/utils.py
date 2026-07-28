import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from src.config import *

def train_lgb_model(X_train, X_val, y_train, y_val):
    params = {
        "objective": "binary",
        "metric": "auc",
        "random_state": RANDOM_SEED,
        "verbosity": -1
    }
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=3000,
        valid_sets=[lgb_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(100)
        ]
    )
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_val, val_pred)
    print(f"验证集AUC: {auc:.6f}")
    return model