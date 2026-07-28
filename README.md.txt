# Bank Churn Prediction – Kaggle Playground Series S4E1

> **Binary classification** to predict whether a bank customer will churn (exit the bank).  
> Achieved **Public LB 0.88779** with a single LightGBM model.

---

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Competition Background](#competition-background)
- [Core Challenges](#core-challenges)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Exploratory Data Analysis – Key Insights](#exploratory-data-analysis--key-insights)
- [Feature Engineering & Modeling](#feature-engineering--modeling)
- [Model Performance](#model-performance)
- [Future Work](#future-work)
- [References & Acknowledgements](#references--acknowledgements)

---

## Project Overview
This project tackles the **Playground Series Season 4, Episode 1** competition on Kaggle.  
The goal is to predict whether a customer will leave the bank (binary target `Exited`) based on demographic and account information.  
The dataset is synthetically generated but mimics real-world banking churn patterns.

---

## Competition Background
- **Source**: [Kaggle Playground Series S4E1](https://www.kaggle.com/competitions/playground-series-s4e1)  
- **Evaluation metric**: Area Under the ROC Curve (ROC‑AUC)  
- **Data size**: 165,034 training rows, 110,023 test rows, 14 features  
- **Note**: All data is synthetic, yet the feature distributions are designed to be close to real banking scenarios.

---

## Core Challenges
- **Class imbalance** – only ~20% of customers churn, requiring careful stratification and metric selection.  
- **Categorical encoding** – `Geography` (3 countries) and `Gender` (2 values) needed proper treatment.  
- **Feature engineering** – creating meaningful interactions (e.g., `Age * NumOfProducts`, balance-to-salary ratio) to improve predictive power.  
- **Overfitting** – using early stopping and validation splits to generalise well on the public leaderboard.

---

## Tech Stack
- **Language**: Python 3.10  
- **Data manipulation**: Pandas, NumPy  
- **Visualisation**: Matplotlib, Seaborn  
- **Machine learning**: Scikit‑learn, LightGBM  
- **Encoding**: Category Encoders (Target Encoding)  
- **Environment**: Jupyter Notebook / Kaggle Notebook

---

## Project Structure
```
Bank-Churn-Prediction-Kaggle-S4E1/
│
├── train.csv                 # Training data
├── test.csv                  # Test data (without target)
├── sample_submission.csv     # Submission format
├── notebook.ipynb            # Main analysis & modeling notebook
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── submission.csv            # Final predictions (output)
```

---

## Quick Start
1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/Bank-Churn-Prediction-Kaggle-S4E1.git
   cd Bank-Churn-Prediction-Kaggle-S4E1
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook** (or execute the Python script)  
   All steps – from EDA to prediction – are contained in `notebook.ipynb`.  
   The final submission file `submission.csv` will be generated.

---

## Exploratory Data Analysis – Key Insights
- **Geography**: Germany has the highest churn rate (~37.9%), followed by Spain (~17.2%) and France (~16.5%).  
- **Gender**: Female customers are nearly twice as likely to churn (~28.0%) compared to males (~15.9%).  
- **Age**: Churners tend to be older (median ~45 vs. ~36 for non-churners).  
- **Balance**: Customers with zero balance have lower churn, while high balance (>100k) correlates with higher churn.  
- **NumOfProducts**: Customers with 1 or 2 products are common; churn is higher for those with 1 product and significantly lower for those with 3 or 4.  
- **IsActiveMember**: Inactive members have a churn rate of ~29.7%, whereas active members churn at only ~12.5%.  
- **Tenure**: Newer customers (tenure 0–2) are more likely to leave; churn decreases with longer tenure.  
- **Correlations**: Age and `NumOfProducts` are negatively correlated (-0.10), while Balance and `NumOfProducts` are more strongly negative (-0.36).

---

## Feature Engineering & Modeling
### Data Preprocessing
- Dropped `id`, `CustomerId`, and `Surname` (non‑predictive).  
- Used **Stratified train/validation split** (80/20) to preserve class proportions.  
- Applied **Target Encoding** for `Geography` and `Gender` – fitted only on training data to avoid leakage.

### Engineered Features
- `ZeroBalance`: binary indicator (Balance == 0).  
- `Age_Product`: interaction between age and number of products.  
- `Balance_Salary`: ratio of balance to estimated salary (clipped to avoid division by zero).

### Model Selection
- **LightGBM** – chosen for its speed, handling of categorical-like features, and built-in early stopping.  
- Hyperparameters: binary objective, AUC metric, 3000 boosting rounds with early stopping (200 rounds patience).  
- Final model achieved a validation AUC of **0.8897**.

---

## Model Performance
| Model | Validation AUC | Public LB Score |
|-------|---------------|-----------------|
| LightGBM (single) | 0.8897 | **0.88779** |
| *Ensemble (planned)* | – | – |

> *Only a single LightGBM was submitted; the score already outperformed many simple baselines.*

---

## Future Work
- **Hyperparameter tuning** – use Optuna or GridSearch to refine learning rate, tree depth, and regularisation.  
- **Ensemble methods** – combine LightGBM with XGBoost and a neural network (e.g., TabNet) via stacking.  
- **More feature engineering** – create tenure‑age interactions, product‑balance clusters, or derive country‑specific statistics.  
- **Deployment** – wrap the model in a simple REST API (Flask/FastAPI) for real‑time predictions.  
- **Interpretability** – apply SHAP to explain model predictions and uncover deeper drivers of churn.

---

## References & Acknowledgements
- Kaggle Competition: [Playground Series S4E1](https://www.kaggle.com/competitions/playground-series-s4e1)  
- Inspiration from public notebooks and discussions on the competition forum.  
- LightGBM documentation: [https://lightgbm.readthedocs.io/](https://lightgbm.readthedocs.io/)  
- Category Encoders library: [https://contrib.scikit-learn.org/category_encoders/](https://contrib.scikit-learn.org/category_encoders/)

---

**Author**: [Your Name]  
**Date**: July 2026  
**License**: MIT
```