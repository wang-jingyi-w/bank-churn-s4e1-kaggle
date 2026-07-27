import matplotlib.pyplot as plt
import seaborn as sns
from src.config import *

# 解决中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_categorical_dist(train_df):
    """类别特征：Geography、Gender流失分布计数图"""
    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    sns.countplot(data=train_df, x="Geography", hue=TARGET, ax=axes[0])
    axes[0].set_title("地区 vs 客户流失标签")
    sns.countplot(data=train_df, x="Gender", hue=TARGET, ax=axes[1])
    axes[1].set_title("性别 vs 客户流失标签")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "cat_dist.png"), dpi=150)
    plt.close()

def plot_boxplot_numeric(train_df):
    """数值特征箱线图（CreditScore/Age/Balance等6张图）"""
    fig, axes = plt.subplots(2, 3, figsize=(18,8))
    axes = axes.flatten()
    for idx, col in enumerate(NUM_COLS):
        sns.boxplot(data=train_df, y=col, x=TARGET, ax=axes[idx])
        axes[idx].set_title(f"{col} 分布 vs 流失")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "num_boxplot.png"), dpi=150)
    plt.close()

def plot_correlation_heatmap(train_df):
    """相关性热力图"""
    corr_df = train_df[NUM_COLS + [TARGET]].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm")
    plt.title("数值特征相关性热力图")
    plt.savefig(os.path.join(FIGURES_DIR, "corr_heatmap.png"), dpi=150)
    plt.close()

def run_all_eda(train_df):
    plot_categorical_dist(train_df)
    plot_boxplot_numeric(train_df)
    plot_correlation_heatmap(train_df)
    print("EDA绘图完成，图片保存至output/figures")