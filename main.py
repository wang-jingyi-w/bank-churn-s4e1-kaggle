#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
银行客户流失预测 - 主程序入口
Usage: python main.py
"""

import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from src.data.make_dataset import load_data, preprocess_data
from src.models.train_model import train_model
from src.models.predict_model import predict, create_submission
from src.visualization.visualize import plot_feature_importance, plot_correlation
from src.config import CONFIG

def main():
    print("="*60)
    print("🏦 银行客户流失预测系统")
    print("="*60)
    
    # 1. 加载数据
    print("\n[1/6] 加载数据...")
    train, test, sub = load_data()
    
    # 2. EDA 可视化 (可选)
    print("\n[2/6] 生成可视化报告...")
    os.makedirs("reports/figures", exist_ok=True)
    
    # 相关性热力图
    fig = plot_correlation(train)
    fig.savefig("reports/figures/correlation_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("✅ 相关性热力图已保存: reports/figures/correlation_heatmap.png")
    
    # 3. 预处理
    print("\n[3/6] 数据预处理...")
    X_train, X_val, X_test, y_train, y_val, encoders = preprocess_data(train, test)
    
    # 4. 训练模型
    print("\n[4/6] 训练 LightGBM 模型...")
    model, auc = train_model(X_train, y_train, X_val, y_val)
    
    # 5. 保存编码器
    print("\n[5/6] 保存编码器...")
    joblib.dump(encoders, "models/encoders.pkl")
    print("✅ 编码器已保存: models/encoders.pkl")
    
    # 6. 生成提交文件
    print("\n[6/6] 生成预测提交文件...")
    predictions = predict(X_test)
    create_submission(predictions)
    
    # 7. 特征重要性 (可选)
    print("\n📊 特征重要性分析:")
    fig = plot_feature_importance(model, X_train.columns.tolist())
    fig.savefig("reports/figures/feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("✅ 特征重要性图已保存: reports/figures/feature_importance.png")
    
    print("\n" + "="*60)
    print(f"✅ 全部完成！最终验证集 AUC: {auc:.6f}")
    print("📁 提交文件: submission.csv")
    print("="*60)

if __name__ == "__main__":
    main()