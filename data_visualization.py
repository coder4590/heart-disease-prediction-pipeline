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

# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
    
# data (as pandas dataframes) 
X = heart_disease.data.features 
Y= heart_disease.data.targets 

def analysis_data():

    
    
    # metadata 
    print(heart_disease.metadata) 
    
    # variable information 
    print(heart_disease.variables)

    print("Little detail of the like of hte data which are we working on")
    print(f"Columns name and type : {X.columns.tolist()}")
    print(f"column name of the target columns : {Y.columns.tolist()}")
    print(f"Data types of the columns : {X.dtypes}")
    print(f"Other type of detail : {X.describe()}")
    print(f"Length of the total data: {len(X)}")

    # 1. Bar plot of column names (just for visual)
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(X.columns)), [1]*len(X.columns))
    plt.yticks(range(len(X.columns)), X.columns)
    plt.title('Column Names')
    plt.xlabel('Has Column')
    plt.tight_layout()
    plt.savefig('Columns.png')
    
    
    # 2. Data types as bar plot
    dtype_counts = X.dtypes.value_counts()
    plt.figure(figsize=(6, 4))
    dtype_counts.plot(kind='bar', color='steelblue')
    plt.title('Data Types Distribution')
    plt.xlabel('Data Type')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('Datatype.png')
    
    
    # 3. Describe as heatmap
    desc = X.describe()
    plt.figure(figsize=(12, 6))
    sns.heatmap(desc.iloc[:, :10], annot=True, cmap='coolwarm', fmt='.1f')
    plt.title('Statistical Description (First 10 Columns)')
    plt.tight_layout()
    plt.savefig('description.png')





    # now we will do some of the analysis on the data and the like of it and then we will jumpt to the liek of the data cleaning and the like of then the like of the feature engineering and the liek fi t and that is the teh like of whole scenoro for it 


    df=X.copy()
    df['num']=Y.values

    plt.figure(figsize=(8,5))
    sns.boxplot(x='num', y='age', data=df)
    plt.title("Age vs Heart Disease")
    plt.xlabel('Diease Chances (0: No disease, (1-4): Disease)')
    plt.ylabel('Age')
    plt.tight_layout()
    plt.savefig('Age_vs_Heart.png')
    plt.show()


    plt.figure(figsize=(8,5))
    pd.crosstab(df['sex'],df['num']).plot(kind='bar')
    plt.title('sex vs Heart Disease')
    plt.xlabel('Gender (0=female,1=Male)')
    plt.ylabel('Heart disease count ')
    plt.legend(title="Disease Severity (num)")
    plt.tight_layout()
    plt.savefig('Sex_vs_Heart Disease')

    # now we will like of the do the same the liek of the with like of hte chest pain type becuase it will also the like of the bceom the ike of hte strong feature so we cannnot skip it and the lie of it 

    plt.figure(figsize=(8,5))
    pd.crosstab(df['cp'], df['num']).plot(kind='bar')
    plt.title('Chest Pain Type vs Heart Disease')
    plt.xlabel('Chest Pain Type')
    plt.ylabel('Count')
    plt.legend(title='Disease Severity (num)')
    plt.tight_layout()
    plt.savefig('cp_vs_heart_disease.png')


    # now we will talked about the like of the heart disase vs the like of hte resting blood pressure which is like of the also the good feature for the lie of the doing and passing the like of the heart deisase and this data clearly see me the like of hte some of the medical data the origanl one 

    plt.figure(figsize=(8,5))
    sns.boxplot(x='num', y='trestbps', data=df)
    plt.title('Resting Blood Pressure vs Heart Disease')
    plt.xlabel('Disease Severity (0=No Disease)')
    plt.ylabel('Blood Pressure (mm Hg)')
    plt.tight_layout()
    plt.savefig('trestbps_vs_disease.png')

    # nwo the like  of the we will see teh like of hte to see if the like of there is the high high colestrol level is the like of hte related to the like of hte 
    plt.figure(figsize=(8,5))
    sns.boxplot(x='num', y='chol', data=df)
    plt.title('Cholesterol vs Heart Disease')
    plt.xlabel('Disease Severity (0=No Disease)')
    plt.ylabel('Cholesterol (mg/dL)')
    plt.tight_layout()
    plt.savefig('chol_vs_disease.png')
    plt.show()

    # this is the like of the plot which is is known as the like of the pie chart and the like of the which we used for the like of hte for the percentage of teh resting blood sugar
    df['has_disease'] = (df['num'] > 0).astype(int)

    # ========== PLOT 1: Pie Chart ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fbs0 = df[df['fbs'] == 0]['has_disease'].value_counts()
    axes[0].pie(fbs0, labels=['No Disease', 'Disease'], autopct='%1.1f%%', 
                colors=['green', 'red'], startangle=90)
    axes[0].set_title('Normal Blood Sugar (fbs=0)')
    fbs1 = df[df['fbs'] == 1]['has_disease'].value_counts()
    axes[1].pie(fbs1, labels=['No Disease', 'Disease'], autopct='%1.1f%%', 
                colors=['green', 'red'], startangle=90)
    axes[1].set_title('High Blood Sugar (fbs=1)')
    plt.suptitle('Fasting Blood Sugar vs Heart Disease', fontsize=14)
    plt.tight_layout()
    plt.savefig('fbs_pie.png')

    # this is the like of teh bar which is showing hte like of hte ecg pattern and the like of it and that is what we are doing right now 
    ecg_disease = df.groupby('restecg')['has_disease'].mean() * 100

    plt.figure(figsize=(6, 4))
    plt.bar(['0', '1', '2'], ecg_disease.values, color=['green', 'orange', 'red'])
    plt.title('ECG vs Heart Disease')
    plt.xlabel('ECG (0=Normal, 1=ST-T, 2=LVH)')
    plt.ylabel('Disease %')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig('restecg_vs_disease.png')

    # this is to show whta pattern has maximum heart has with the like of the heart disease 

    avg_thalach = df.groupby('num')['thalach'].mean()

    plt.figure(figsize=(7, 5))
    plt.bar([str(i) for i in avg_thalach.index], avg_thalach.values, 
            color=['green', 'lightcoral', 'coral', 'red', 'darkred'])
    plt.title('Average Max Heart Rate vs Disease')
    plt.xlabel('Disease Severity (0=No Disease)')
    plt.ylabel('Average Max Heart Rate (bpm)')
    plt.tight_layout()
    plt.savefig('thalach_vs_disease.png')

    # now we will work with the like of hte exercise induxed angine 
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))


    ex0 = df[df['exang'] == 0]['has_disease'].value_counts()
    axes[0].pie(ex0, labels=['No Disease', 'Disease'], autopct='%1.1f%%', colors=['green', 'red'])
    axes[0].set_title('No Exercise Angina (0)')

    # Pie for exang=1
    ex1 = df[df['exang'] == 1]['has_disease'].value_counts()
    axes[1].pie(ex1, labels=['No Disease', 'Disease'], autopct='%1.1f%%', colors=['green', 'red'])
    axes[1].set_title('Has Exercise Angina (1)')

    plt.tight_layout()
    plt.savefig('exang_pie_plot.png')

    # this is the like of hte correlation heatmap and the like of it so that we can find the like of hte pattern for it 

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='RdBu_r', center=0, fmt='.2f')
    plt.title('Correlation Heatmap - All Features vs num')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
