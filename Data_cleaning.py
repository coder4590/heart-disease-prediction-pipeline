import pandas as pd
import numpy as np
import csv
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, RandomTreesEmbedding
from sklearn.preprocessing import StandardScaler , LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split,KFold, cross_val_score, RandomizedSearchCV
from scipy.stats import randint, uniform 
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from ucimlrepo import fetch_ucirepo 

def clean_data():

    df = X.copy()
    df['num'] = Y.values


    print("To see if how many of the values are missing ")
    print(f"Missing Values: {X.isnull().sum()}")


    print("to check the duplicate values")
    print(f"Duplicate Values Check : {X.duplicated().sum()}")
    " so there is not values such as the which are the duplicated or the like of hte such of "
    "row which need to be removed for the duplicated but still for prevention we will drop_duplicagte"
    df=df.drop_duplicates()





    print("IMPOSSIBLE VALUES CHECK")


    # 1. Age - should be between 0 and 120
    print("\n1. Age (valid: 0-120):")
    print(f"   Min: {df['age'].min()}, Max: {df['age'].max()}")
    print(f"   Impossible: {(df['age'] < 0).sum() + (df['age'] > 120).sum()}")

    # 2. Sex - should be 0 or 1
    print("\n2. Sex (valid: 0,1):")
    print(f"   Values: {df['sex'].unique()}")
    print(f"   Impossible: {((df['sex'] != 0) & (df['sex'] != 1)).sum()}")

    # 3. cp - should be 0,1,2,3
    print("\n3. Chest Pain (valid: 0,1,2,3):")
    print(f"   Values: {sorted(df['cp'].unique())}")
    print(f"   Impossible: {((df['cp'] < 0) | (df['cp'] > 3)).sum()}")

    # 4. trestbps - resting BP: 80-250 realistic
    print("\n4. Resting BP (realistic: 80-250):")
    print(f"   Min: {df['trestbps'].min()}, Max: {df['trestbps'].max()}")
    print(f"   Too low (<80): {(df['trestbps'] < 80).sum()}")
    print(f"   Too high (>250): {(df['trestbps'] > 250).sum()}")

    # 5. chol - cholesterol: 100-600 realistic
    print("\n5. Cholesterol (realistic: 100-600):")
    print(f"   Min: {df['chol'].min()}, Max: {df['chol'].max()}")
    print(f"   Too low (<100): {(df['chol'] < 100).sum()}")
    print(f"   Too high (>600): {(df['chol'] > 600).sum()}")

    # 6. fbs - should be 0 or 1
    print("\n6. Fasting Blood Sugar (valid: 0,1):")
    print(f"   Values: {df['fbs'].unique()}")

    # 7. restecg - should be 0,1,2
    print("\n7. Resting ECG (valid: 0,1,2):")
    print(f"   Values: {sorted(df['restecg'].unique())}")

    # 8. thalach - max HR: 60-250 realistic
    print("\n8. Max Heart Rate (realistic: 60-250):")
    print(f"   Min: {df['thalach'].min()}, Max: {df['thalach'].max()}")
    print(f"   Too low (<60): {(df['thalach'] < 60).sum()}")
    print(f"   Too high (>250): {(df['thalach'] > 250).sum()}")

    # 9. exang - should be 0 or 1
    print("\n9. Exercise Angina (valid: 0,1):")
    print(f"   Values: {df['exang'].unique()}")

    # 10. oldpeak - should be >= 0 (ST depression can't be negative)
    print("\n10. Oldpeak (valid: >= 0):")
    print(f"    Min: {df['oldpeak'].min()}, Max: {df['oldpeak'].max()}")
    print(f"    Negative values: {(df['oldpeak'] < 0).sum()}")

    # 11. slope - should be 0,1,2
    print("\n11. Slope (valid: 0,1,2):")
    print(f"    Values: {sorted(df['slope'].unique())}")

    # 12. ca - should be 0,1,2,3,4
    print("\n12. CA (valid: 0,1,2,3):")
    print(f"    Values: {sorted(df['ca'].dropna().unique())}")

    # 13. thal - should be 0,1,2,3
    print("\n13. Thal (valid: 0,1,2,3):")
    print(f"    Values: {sorted(df['thal'].dropna().unique())}")

    # 14. num - target: 0,1,2,3,4
    print("\n14. Target num (valid: 0,1,2,3,4):")
    print(f"    Values: {sorted(df['num'].unique())}")

    print("Outlier checkers ")

    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Calculate how many subplots needed
    n_features = len(numerical_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for idx, col in enumerate(numerical_features):
        axes[idx].boxplot(df[col])
        axes[idx].set_title(f'{col} - Outlier Check')
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        print(f"{col}: {len(outliers)} outliers detected")

    # Hide unused subplots
    for idx in range(len(numerical_features), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig("Outlier_checker.png")

    df = df.drop(columns=['ca', 'thal'])

    df = df.dropna()

    df = df.reset_index(drop=True)
    
    print(f"\nFinal cleaned shape: {df.shape}")
    print(f"Remaining columns: {df.columns.tolist()}")
    
    return df

