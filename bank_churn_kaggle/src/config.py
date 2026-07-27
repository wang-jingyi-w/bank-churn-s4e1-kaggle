import os

# 全局随机种子，保证可复现
RANDOM_SEED = 42
TEST_SPLIT_RATIO = 0.2

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# 创建文件夹
os.makedirs(FIGURES_DIR, exist_ok=True)

# 文件名
TRAIN_CSV = os.path.join(DATA_RAW_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_RAW_DIR, "test.csv")
SUBMIT_CSV = os.path.join(OUTPUT_DIR, "submission.csv")

# 特征列表
NUM_COLS = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
BINARY_COLS = ["HasCrCard", "IsActiveMember"]
CAT_COLS = ["Geography", "Gender"]
DROP_COLS = ["id", "CustomerId", "Surname"]
TARGET = "Exited"