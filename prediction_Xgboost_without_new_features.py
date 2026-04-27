
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
def predict_model():
    df=clean_data()

    target='num'
    removed_feature=[target]
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

    feature_importance = search.best_estimator_.feature_importances_
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
