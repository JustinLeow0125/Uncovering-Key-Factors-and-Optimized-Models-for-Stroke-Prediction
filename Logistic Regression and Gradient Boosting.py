# ---------------------------------------------------# ---------------------------------------------------# ---------------------------------------------------
# DATA PREPROCESSING
# ---------------------------------------------------# ---------------------------------------------------# ---------------------------------------------------

# ---------------------------------------------------
# IMPORT LIBRARIES
# ---------------------------------------------------

import pandas as pd
import numpy as np

# ---------------------------------------------------
# Load Data
# ---------------------------------------------------

stroke_data = pd.read_csv("C:/Users/swtan/OneDrive - Sunway Education Group/BIS/Final/healthcare-dataset-stroke-data.csv", na_values="N/A")

stroke_data = stroke_data.drop(columns=["id"])

# ---------------------------------------------------
# DATA TRANSFORMATION
# ---------------------------------------------------

# Converting 'hypertension', 'heart_disease', and 'stroke' columns to categorical
stroke_data['hypertension'] = stroke_data['hypertension'].astype('category')
stroke_data['heart_disease'] = stroke_data['heart_disease'].astype('category')
stroke_data['stroke'] = stroke_data['stroke'].astype('category')

# Recoding Binary variables to 1 and 0
stroke_data['ever_married'] = stroke_data['ever_married'].apply(lambda x: 1 if x == "Yes" else 0).astype('category') # 1 for Yes 0 for No 
stroke_data['gender'] = stroke_data['gender'].apply(lambda x: 1 if x == "Male" else 0).astype('category') # 1 for Male 0 for Female
stroke_data['Residence_type'] = stroke_data['Residence_type'].apply(lambda x: 1 if x == "Urban" else 0).astype('category') # 1 for Urban 0 for Rural

# Replacing 'Unknown' in 'smoking_status' with Missing Values
stroke_data['smoking_status'] = stroke_data['smoking_status'].replace("Unknown", np.nan)

# Recoding 'work_type' into numbers and convert to categorical
work_type_mapping = {
    "Private": 0,
    "Self-employed": 1,
    "Govt_job": 2,
    "children": 3,
    "Never_worked": 4
}
stroke_data['work_type'] = stroke_data['work_type'].map(work_type_mapping).astype('category')

# Recoding 'smoking_status' into numbers and convert to categorical
smoking_status_mapping = {  
    "formerly smoked": 0,
    "never smoked": 1,
    "smokes": 2
}
stroke_data['smoking_status'] = stroke_data['smoking_status'].map(smoking_status_mapping).astype('category')

# ---------------------------------------------------
# MISSING DATA HANDLING
# ---------------------------------------------------

# Calculating n (total count) and nmiss (missing count) for each column
n = stroke_data.notnull().sum()
nmiss = stroke_data.isnull().sum()

# Combining into a data frame and viewing
result = pd.DataFrame({
    'Column': stroke_data.columns,
    'Total': n,
    'Missing': nmiss
})

# 'bmi' - Median Imputation
stroke_data['bmi'] = pd.to_numeric(stroke_data['bmi'], errors='coerce')
median_bmi = stroke_data['bmi'].median(skipna=True)
stroke_data['bmi'].fillna(median_bmi, inplace=True)

# 'smoking_status' - Mode Imputation
# Calculating the mode of smoking_status
mode_smoking = stroke_data['smoking_status'].mode()[0]

# Imputing missing values in smoking_status with the mode
stroke_data['smoking_status'].fillna(mode_smoking, inplace=True)

# ---------------------------------------------------
# OUTLIER HANDLING
# ---------------------------------------------------

# IQR calculation
Q1 = stroke_data['bmi'].quantile(0.25)
Q3 = stroke_data['bmi'].quantile(0.75)
IQR = Q3 - Q1

# Defining lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Capping the outliers by replacing outliers with their closest boundary
stroke_data['bmi'] = stroke_data['bmi'].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))


# ---------------------------------------------------# ---------------------------------------------------# ---------------------------------------------------
# Logistic Regression
# ---------------------------------------------------# ---------------------------------------------------# ---------------------------------------------------

# ---------------------------------------------------
# IMPORT LIBRARIES
# ---------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

# ---------------------------------------------------
# LOAD AND PREPROCESS DATA
# ---------------------------------------------------

# Load the dataset
file_path = r"C:\Users\notth\OneDrive - Sunway Education Group\YR 3 SEM 2\BIS\assignment\dataset\stroke_data.csv"
stroke_data = pd.read_csv(file_path)

# Select only relevant variables
relevant_columns = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
                    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']
stroke_data = stroke_data[relevant_columns]

# Treat categorical variables as categories
stroke_data['gender'] = stroke_data['gender'].astype('category')
stroke_data['work_type'] = stroke_data['work_type'].astype('category')
stroke_data['Residence_type'] = stroke_data['Residence_type'].astype('category')

# One-hot encode categorical variables
stroke_data = pd.get_dummies(stroke_data, columns=['gender', 'work_type', 'Residence_type'], drop_first=True)

# ---------------------------------------------------
# TRAIN-TEST SPLIT
# ---------------------------------------------------

# Features and target
X = stroke_data.drop(columns=['stroke'])
y = stroke_data['stroke']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------
# CLASS WEIGHTS COMPUTATION
# ---------------------------------------------------

# Compute class weights based on the training data
classes = np.array([0, 1])  # Convert to NumPy array
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)

# Convert to dictionary format for logistic regression
weights = {0: class_weights[0], 1: class_weights[1]}
print(f"Class Weights: {weights}")

# ---------------------------------------------------
# TRAIN LOGISTIC REGRESSION
# ---------------------------------------------------

# Train weighted logistic regression
lr = LogisticRegression(class_weight=weights, random_state=42, max_iter=1000)
lr.fit(X_train, y_train)

# ---------------------------------------------------
# PREDICTIONS AND EVALUATION
# ---------------------------------------------------

# Predictions
threshold = 0.9  # Decision threshold for Stroke
y_pred_proba_lr = lr.predict_proba(X_test)[:, 1]
y_pred_lr = (y_pred_proba_lr >= threshold).astype(int)

# Evaluation Metrics
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba_lr))
print("Classification Report:")
print(classification_report(y_test, y_pred_lr, target_names=["No Stroke", "Stroke"]))

# Compute and display accuracy at the threshold as a percentage
accuracy = accuracy_score(y_test, y_pred_lr)
print(f"Accuracy at Threshold = {threshold}: {accuracy * 100:.2f}%")

# ---------------------------------------------------
# PLOT ROC CURVE
# ---------------------------------------------------

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_lr)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# ---------------------------------------------------# ---------------------------------------------------# ---------------------------------------------------
# XGBoost
# ---------------------------------------------------# ---------------------------------------------------# ---------------------------------------------------

# -------------------------------------------
# IMPORT LIBRARIES
# -------------------------------------------
import pandas as pd
import numpy as np

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')  # Hide warnings (optional)

# -------------------------------------------
# SET RANDOM SEED FOR REPRODUCIBILITY
# -------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# -------------------------------------------
# LOAD DATA
# -------------------------------------------
file_path = r"C:\Users\notth\OneDrive - Sunway Education Group\YR 3 SEM 2\BIS\assignment\dataset\stroke_data.csv"
stroke_data = pd.read_csv(file_path)

print("Initial data shape:", stroke_data.shape)
print(stroke_data.head(), "\n")

# -------------------------------------------
# CHECK DATA TYPES
# -------------------------------------------
print("Initial data types:\n", stroke_data.dtypes, "\n")

# -------------------------------------------
# CONVERT COLUMNS TO CATEGORY (FOR REFERENCE)
# -------------------------------------------
cat_cols = ['work_type', 'smoking_status']
for col in cat_cols:
    stroke_data[col] = stroke_data[col].astype('category')

print("After casting to category:\n", stroke_data.dtypes, "\n")

# -------------------------------------------
# LABEL-ENCODE CATEGORICAL COLUMNS
# -------------------------------------------
le_work = LabelEncoder()
stroke_data['work_type'] = le_work.fit_transform(stroke_data['work_type'])

le_smoke = LabelEncoder()
stroke_data['smoking_status'] = le_smoke.fit_transform(stroke_data['smoking_status'])

print("After label encoding:\n", stroke_data.dtypes, "\n")

# -------------------------------------------
# ADDRESS CLASS IMBALANCE USING CLASS WEIGHTS
# -------------------------------------------
class_weights_stroke = compute_class_weight(
    class_weight='balanced',
    classes=stroke_data['stroke'].unique(),
    y=stroke_data['stroke']
)
stroke_weights = {cls: weight for cls, weight in zip(stroke_data['stroke'].unique(), class_weights_stroke)}
stroke_ratio = class_weights_stroke[0] / class_weights_stroke[1]

print("Stroke class weights:", stroke_weights)
print(f"Stroke ratio (for scale_pos_weight): {stroke_ratio:.2f}", "\n")

# -------------------------------------------
# CORRELATION MATRIX
# -------------------------------------------
correlation_matrix = stroke_data.corr(numeric_only=True)
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix")
plt.show()

# -------------------------------------------
# DEFINE FEATURES (X) AND TARGET (y)
# -------------------------------------------
X = stroke_data.drop(columns=['stroke'])
y = stroke_data['stroke']

# -------------------------------------------
# HYPERPARAMETER TUNING WITH GRIDSEARCHCV
# -------------------------------------------
param_grid = {
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [50, 100],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'scale_pos_weight': [stroke_ratio]  # use our computed ratio
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

xgb_estimator = XGBClassifier(
    eval_metric='logloss',
    random_state=RANDOM_SEED
)

grid_search = GridSearchCV(
    estimator=xgb_estimator,
    param_grid=param_grid,
    scoring='accuracy',
    cv=cv,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X, y)

print("\nBest Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_, "\n")

# -------------------------------------------
# TRAIN-TEST SPLIT
# -------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y
)

# Train the model with the best found parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# -------------------------------------------
# PREDICT PROBABILITIES AND CALCULATE AUC
# -------------------------------------------
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score: {roc_auc:.6f}")

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', label=f'ROC AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# -------------------------------------------
# APPLY CUSTOM THRESHOLD AND EVALUATE
# -------------------------------------------
custom_threshold = 0.9
y_pred = (y_pred_proba >= custom_threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy at threshold = {custom_threshold}: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ---------------------------------------------------# ---------------------------------------------------# ---------------------------------------------------
# Feature Importance and Logistic Regression Coefficients
# ---------------------------------------------------# ---------------------------------------------------# ---------------------------------------------------

# ---------------------------------------------------
# IMPORT LIBRARIES
# ---------------------------------------------------
import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ---------------------------------------------------
# LOAD DATASET AND DEFINE VARIABLES
# ---------------------------------------------------

# Load the dataset
data = pd.read_csv(r"C:\Users\notth\OneDrive - Sunway Education Group\YR 3 SEM 2\BIS\dataset\stroke_data.csv")

# Define features and target variable
X = data.drop(columns=['stroke'])  # Features
y = data['stroke']  # Target variable

# ---------------------------------------------------
# TRAIN-TEST SPLIT
# ---------------------------------------------------

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------------------------------------------
# XGBOOST MODEL AND FEATURE IMPORTANCE
# ---------------------------------------------------

# Train an XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Get feature importance from XGBoost
xgb_importance = xgb_model.feature_importances_
xgb_features = X.columns

# Plot feature importance for XGBoost
plt.figure(figsize=(10, 6))
plt.barh(xgb_features, xgb_importance, color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('XGBoost Feature Importance')
plt.show()

# ---------------------------------------------------
# LOGISTIC REGRESSION AND COEFFICIENTS
# ---------------------------------------------------

# Train a Logistic Regression model
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)

# Get feature coefficients from Logistic Regression
logreg_coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': logreg.coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("Logistic Regression Coefficients:")
print(logreg_coefficients)

# End of Scripts