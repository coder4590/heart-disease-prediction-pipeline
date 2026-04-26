# heart-disease-prediction-pipeline
# ❤️ The Silent Killer — First-Time Heart Disease Prediction

**A story about data, limitation, and the pursuit of honest prediction.**

---

## The Man Who Felt Fine

There is a man. 45 years old. Works a desk job. Eats whatever he wants. Sometimes feels a little breathless after climbing two floors, but he calls it "getting old."

One day, he visits a clinic. The doctor runs a few basic tests:

- Blood pressure check — 5 minutes
- Cholesterol test — quick blood draw
- ECG while resting — electrodes on the chest, painless
- Treadmill walk — "tell me if your chest hurts"
- Some questions about chest pain — "describe what you feel"

Nothing invasive. No needles in arteries. No nuclear dye. No catheter up the groin. Just simple, routine, first-visit tests.

**And then the question comes:**

> *"Doctor, do I have heart disease?"*

This project answers that question. With **79% accuracy**, using nothing but those basic tests.

---

## The Problem With Most Heart Disease Models

Go search "heart disease prediction" on Kaggle. You will find hundreds of notebooks claiming 95%, 98%, even 99% accuracy.

**They are all lying to you.**

Not intentionally. But they include features like `ca` (fluoroscopy vessel count) and `thal` (thallium stress test). These are tests you only do when:

- You already have severe symptoms
- A doctor already strongly suspects heart disease
- You are already on the operating table, basically

Using these features to "predict" heart disease is like predicting rain by looking out the window and seeing water falling from the sky. It is not prediction. It is observation.

**This project removes those features. Completely.**

---

## What We Dropped (And Why)

| Feature | What It Is | Why We Removed It |
|---|---|---|
| `ca` | Number of vessels colored by fluoroscopy | Requires catheter injection into heart arteries |
| `thal` | Thallium stress test result | Nuclear scan — only for confirmed cardiac patients |

These are **post-diagnosis** procedures. You don't do them on a healthy person walking into a clinic for the first time. Including them is data leakage — the model cheats by seeing the answer before making its prediction.

**Our model works with features available BEFORE any invasive test.**

---

## The Data — And Its Brutal Limitation

Let me be honest with you.

This dataset has **303 rows**. Three hundred and three.

Not 10,000. Not 100,000. Three hundred and three.

In machine learning, this is tiny. Micro. Almost microscopic.

And the target variable `num` has 5 classes:
- 0 = No disease (~160 patients)
- 1 = Mild disease (~55 patients)
- 2 = Moderate disease (~35 patients)
- 3 = Severe disease (~35 patients)
- 4 = Very severe (~13 patients)

**Thirteen patients in class 4.** How can any model learn from thirteen examples?

Answer: It cannot.

This is not a failure of the model. This is a failure of the data. And as a machine learning engineer, you must learn to accept this. Some data is just limited. Some problems cannot be solved with perfect accuracy because the information simply isn't there.

**We chose binary classification** — disease (1-4) vs no disease (0). This is the honest approach. This is what the data can actually support.

---

## The Visual Story — Understanding Each Feature

### 1. Age vs Heart Disease

<img width="800" height="500" alt="Age vs Heart" src="https://github.com/user-attachments/assets/587be0bb-8e71-434c-a7eb-775ed90c892f" />


**What you see:** The boxes for diseased and non-diseased patients overlap almost completely. The median age is around 55 for both groups.

**What it means:** Age alone tells you almost nothing. A 40-year-old can have severe disease. A 65-year-old can have clean arteries. Do not assume youth protects you. Do not assume age condemns you.

---

### 2. Sex vs Heart Disease

<img width="640" height="480" alt="Sex vs Heart Disease" src="https://github.com/user-attachments/assets/f59c766f-328b-4aec-8ca6-2c37da52fff2" />


**What you see:** In females, the majority are in class 0 (no disease). In males, the disease categories dominate — 46 in class 1, 30 in class 2, 28 in class 3, 12 in class 4.

**What it means:** Men face significantly higher cardiac risk. This is not opinion — this is the data screaming it. For every woman with severe disease, there are roughly four men.

---

### 3. Chest Pain Type vs Heart Disease

<img width="640" height="480" alt="cp_vs_heart_disease" src="https://github.com/user-attachments/assets/9a2c472d-2acb-459c-b529-c8ca636d3a74" />


**What you see:** Type 4 (asymptomatic — no chest pain at all) has the HIGHEST concentration of disease cases. Look at the bar — it towers over the others in disease count.

**What it means:** The most dangerous chest pain is NO chest pain. Silent ischemia kills without warning. Patients who feel nothing are often the ones with the worst blockages. This feature became the #1 predictor in our model (33% importance).

---

### 4. Resting Blood Pressure vs Disease

<img width="800" height="500" alt="trestbps_vs_disease" src="https://github.com/user-attachments/assets/8dba04bf-8415-4603-8f05-50fd65a1e036" />


**What you see:** A gradual upward shift. Class 3 and 4 patients cluster at higher blood pressure levels (150-180 mm Hg). The healthy group has lower medians.

**What it means:** High blood pressure damages arteries over years. It alone is not the killer, but combined with other factors, it accelerates the path to disease.

---

### 5. Cholesterol vs Disease

<img width="800" height="500" alt="chol_vs_disease" src="https://github.com/user-attachments/assets/e908a0cf-132d-478a-8b38-c99908d2706f" />


**What you see:** Almost no pattern. The boxes are nearly identical across all disease levels. Outliers with 400+ cholesterol exist in BOTH healthy and diseased groups.

**What it means:** This is the most misunderstood feature. Cholesterol alone does not predict heart disease. It is one ingredient in a complex recipe. A person with high cholesterol but no other risk factors can be perfectly healthy. Context matters.

---

### 6. ECG Results vs Disease

<img width="600" height="400" alt="restecg_vs_disease" src="https://github.com/user-attachments/assets/e345f968-2803-4bd3-b3fa-a5ee45f09388" />


**What you see:** A stair-step pattern. Normal ECG (0) has the lowest disease percentage. ST-T abnormality (1) is higher. LVH (2) is the highest.

**What it means:** The ECG is a direct electrical reading of your heart. If it shows abnormalities, listen. Especially LVH — it means your heart muscle has already thickened from years of strain. This is a strong, reliable predictor.

---

### 7. Max Heart Rate vs Disease

<img width="700" height="500" alt="thalach_vs_disease" src="https://github.com/user-attachments/assets/59626de9-2b45-4bf8-bb4c-22a0758a1503" />


**What you see:** The bars drop steadily. Healthy patients reach ~150 bpm on average. Severe disease patients barely reach ~120 bpm.

**What it means:** A sick heart cannot perform. It hits a lower ceiling. This negative correlation — lower heart rate = higher disease — makes it one of our strongest features.

---

### 8. Correlation Heatmap

<img width="1000" height="800" alt="correlation_heatmap" src="https://github.com/user-attachments/assets/39d3af1e-fc44-4c03-8c37-0bf17554b4a7" />


**What you see:** Look at the bottom rows (`num` and `has_disease`). The strongest correlations belong to `cp` (0.41), `thalach` (-0.42), `exang` (0.39), `oldpeak` (0.41), and `slope` (0.34). The weakest are `chol` (0.07) and `fbs` (0.06).

**What it means:** The model listens to symptoms and stress test results. Exactly what a cardiologist would prioritize. Cholesterol and blood sugar, despite their reputation, are nearly useless as standalone predictors in this dataset.

---

### 9. Feature Importance (After Engineering)

<img width="1000" height="600" alt="Feature_importance" src="https://github.com/user-attachments/assets/c1ed3471-9dbf-49f7-a54b-7004fb329405" />


**What you see:** `cp` dominates at 33%. `exang` follows at 20%. Engineered features like `oldpeak_cat_2`, `thalach_bin_2`, and `bp_cat_3` appear in the top 15 — proving our engineering added value.

**What it means:** Chest pain type and exercise-induced angina are the strongest signals. A patient who has asymptomatic chest pain AND gets angina during exercise is extremely likely to have heart disease. The model has learned what doctors already know.

---

## The Model — Finding the Best Algorithm for Small Data

We tested two champions:

| Model | Accuracy | Why |
|---|---|---|
| **Random Forest** | **0.79** | Bagging + shallow trees = handles small data well |
| XGBoost | 0.75 | Boosting amplifies noise when data is scarce |

**Winner: Random Forest.**

This surprised us. XGBoost is the king of structured data. But with only 303 rows, its boosting strategy backfires — it aggressively learns noise instead of signal.

Random Forest, with its bagging approach and ensemble of shallow trees, is more forgiving. It doesn't overthink. It doesn't overfit. It just works.

**The lesson:** The best algorithm is not the most popular one. It is the one that fits your data.

---

## The Pipeline — Bulletproof Design

We built a production-grade pipeline:

1. **Data Cleaning** — Remove duplicates, check impossible values, detect outliers
2. **Leakage Removal** — Drop `ca` and `thal` before any training
3. **Feature Engineering** — Bin continuous variables (age, cholesterol, BP, heart rate) into categories that capture non-linear relationships
4. **ColumnTransformer** — StandardScaler for numeric, OneHotEncoder for categorical — no leakage between train and test
5. **Hyperparameter Tuning** — RandomizedSearchCV with 3-fold cross-validation (small data = fewer folds)
6. **Evaluation** — Accuracy, precision, recall, confusion matrix, feature importance

Every step protects against overfitting. Every decision is justified by the data size.

---

## 🛠️ Tech Stack

| Category | Tools & Libraries |
|---|---|
| **Language** | Python 3.8+ |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Preprocessing** | Scikit-learn (StandardScaler, OneHotEncoder, ColumnTransformer) |
| **Models** | Scikit-learn (RandomForestClassifier), XGBoost (XGBClassifier) |
| **Pipeline** | Scikit-learn Pipeline |
| **Hyperparameter Tuning** | RandomizedSearchCV, K-Fold Cross-Validation |
| **Evaluation** | accuracy_score, classification_report, confusion_matrix, feature_importances_ |
| **Statistical Analysis** | SciPy (randint, uniform distributions) |
| **Data Source** | UCI Machine Learning Repository (Heart Disease Dataset) |

---

## The Honest Truth

This model achieves **79% accuracy** on predicting whether a first-time patient has heart disease — using only basic, non-invasive tests.

Is 79% perfect? No.

Is it better than a coin flip? Yes — massively.

Is it useful for a clinic? Absolutely — a doctor can run these tests in 30 minutes and get a 79% accurate risk assessment before deciding on further procedures.

79% is not 99%. But 99% models are usually cheaters. This model is honest. It works with what is actually available at the first visit.

---

## Data Limitations — Accepting Reality

Machine learning engineers love to chase accuracy. But sometimes, the data simply does not have more to give.

303 rows. 5 classes. 13 patients in the rarest category.

You cannot manifest accuracy from nothing. You cannot engineer your way past a fundamental data shortage. The limitation is real. Accept it. Work within it. Deliver the best model the data can support — and be honest about what that is.

---

## How To Run

```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
pip install -r requirements.txt
python predict_model.py
