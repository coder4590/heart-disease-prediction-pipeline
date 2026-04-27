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

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

heart_disease = fetch_ucirepo(id=45) 
    
# data (as pandas dataframes) 
X = heart_disease.data.features 
Y= heart_disease.data.targets 

def clean_data(X,Y):

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

def new_feature(df):

    # for col in ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope']:
    #     print(f"\n{'='*40}")
    #     print(f"FEATURE: {col}")
    #     print(f"{'='*40}")
    #     print(f"Unique values: {df[col].nunique()}")
    #     print(f"Value counts:\n{df[col].value_counts().sort_index()}")
    #     print(f"Min: {df[col].min()}, Max: {df[col].max()}, Mean: {df[col].mean():.2f}")
    
    df['age_bin'] = pd.cut(df['age'], bins=[29, 45, 60, 77], labels=[0, 1, 2], include_lowest=True)
    df['age_bin'] = df['age_bin'].astype(int)  # feature one 1 and we will keep both of the original age and the liek of the this new feature

    df['bp_cat'] = pd.cut(df['trestbps'], bins=[0, 120, 130, 140, 250], labels=[0, 1, 2, 3])
    df['bp_cat'] = df['bp_cat'].astype(int)  # feature one 2 and this is thel ike of hte we will drop the like of the so the like of the so the like of the so 

    df['chol_cat'] = pd.cut(df['chol'], bins=[0, 200, 240, 600], labels=[0, 1, 2])
    df['chol_cat'] = df['chol_cat'].astype(int)  # feature number 3 we will drop the like of teh original feature and the like of the thing about it 

    df['thalach_bin'] = pd.cut(df['thalach'], bins=[0, 120, 160, 250], labels=[0, 1, 2])
    df['thalach_bin'] = df['thalach_bin'].astype(int) # feature number 4 and we will keep the both of the original and thel ike of the other type of featureand the like of it 

    df['oldpeak_cat'] = pd.cut(df['oldpeak'], bins=[-0.1, 1, 2, 7], labels=[0, 1, 2])
    df['oldpeak_cat'] = df['oldpeak_cat'].astype(int) # feature number 5 and we will keep the both of the original and the like of the as we well as of the other type of the feature 

    return df


def predict_data():
    df=clean_data(X,Y)
    df=new_feature(df)

    df['num'] = (df['num'] > 0).astype(int)
    
    target='num'
    removed_feature=[target,'trestbps', 'chol']
    numerical_features=df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object','category']).columns.tolist()

    final_numerical_col=[col for col in numerical_features if col not in removed_feature]
    final_categorical_col=[col for col in categorical_features if col not in removed_feature]


    x=df[final_numerical_col+ final_categorical_col]
    y=df[target]

    x_train,x_test,y_train,y_test=train_test_split(
        x,y,
        test_size=0.3,
        random_state=42
    )

    # now we will use the like of hte column transformer and piple line for the like of the building the structure of the thing about it 

    preprocessor=ColumnTransformer(
        [
            ('num', StandardScaler(), final_numerical_col),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), final_categorical_col)
        ]
    )

    pipelines=Pipeline(
        [
            ('pre', preprocessor),
            ('model', xgb.XGBClassifier(random_state=42))
        ]
    )

    param_dist = {
    'model__n_estimators': randint(50,200),    
    'model__learning_rate': uniform(0.01, 0.2),    
    'model__max_depth': randint(2, 6),
    'model__subsample': uniform(0.6, 0.4),
    'model__colsample_bytree': uniform(0.6, 0.4),
    'model__reg_lambda': uniform(0, 20),
    'model__reg_alpha': uniform(0, 5),
    'model__gamma': uniform(0, 5),
    'model__min_child_weight': randint(3, 15)
}

    # Randomized search
    search = RandomizedSearchCV(
        estimator=pipelines,
        param_distributions=param_dist,
        n_iter=30,  # Increase if you have time
        cv=3, # using the cross validation with the 5 cross cv 
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    search.fit(x_train,y_train)

    print("Best Parameters:", search.best_params_)
    print("Best CV Accuracy:", search.best_score_)


    y_pred=search.best_estimator_.predict(x_test)

    
    accuracy=accuracy_score(y_test, y_pred)


    print(f"accuracy for xg: {accuracy:.3f}")
    print(f"Correct_prediction: {accuracy*100:.1f}")

    print("Classification Report")
    print(classification_report(y_test,y_pred))

    print("\nCONFUSION MATRIX:")
    print(confusion_matrix(y_test, y_pred))

    feature_importance = search.best_estimator_.named_steps['model'].feature_importances_
    top_features = pd.DataFrame({
    'feature': x_train.columns,
    'importance': feature_importance
    }).sort_values('importance', ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('Importance')
    plt.title('Top 15 Most Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('Feature_importance.png')



predict_data()
