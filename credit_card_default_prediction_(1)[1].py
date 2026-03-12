# ============================================================
#   CREDIT CARD DEFAULT PREDICTION
#   Pragyan AI Workshop - Day 3 Project
#   Difficulty: Beginner | Type: Classification
# ============================================================
#
#  PROBLEM: Predict whether a customer will DEFAULT (fail to pay)
#           their credit card next month.
#
#  DATASET: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
#           Download the CSV and place it in the same folder as this script.
#           Rename it to: credit_card_default.csv
#
#  HOW TO RUN:
#    1. Install libraries:  pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn
#    2. Download dataset from Kaggle and rename to: credit_card_default.csv
#    3. Run:  python credit_card_default_prediction.py
# ============================================================


# ────────────────────────────────────────────────────────────
# STEP 0: Import Libraries
# ────────────────────────────────────────────────────────────
# pandas  → for loading and manipulating data (like Excel in Python)
# numpy   → for math operations
# sklearn → machine learning library (models, metrics, splitting data)
# xgboost → powerful tree-based model
# imblearn→ handles class imbalance using SMOTE
# matplotlib / seaborn → for charts and graphs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')  # Hide unnecessary warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


# ────────────────────────────────────────────────────────────
# STEP 1: Load the Dataset
# ────────────────────────────────────────────────────────────
print("=" * 60)
print("  CREDIT CARD DEFAULT PREDICTION")
print("=" * 60)
print("\n[STEP 1] Loading dataset...")

try:
    # Read the CSV file into a DataFrame (like a table/spreadsheet)
    df = pd.read_csv("credit_card_default.csv")
    print(f"  ✅ Dataset loaded successfully!")
    print(f"  📊 Shape: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print("\n  ❌ ERROR: File 'credit_card_default.csv' not found!")
    print("  Please download from Kaggle and place in the same folder.")
    print("  URL: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset")
    exit()

# Show first 5 rows so we can see what our data looks like
print("\n  First 5 rows of the dataset:")
print(df.head())


# ────────────────────────────────────────────────────────────
# STEP 2: Understand the Data (EDA - Exploratory Data Analysis)
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[STEP 2] Exploring the Data...")
print("=" * 60)

# Show basic statistics (mean, min, max, etc.)
print("\n  Dataset Info:")
print(df.info())

print("\n  Missing values per column:")
print(df.isnull().sum())

# Our TARGET column - what we want to predict
# 0 = No default (paid on time)
# 1 = Default (did NOT pay)
target_col = 'default.payment.next.month'

# How many customers defaulted vs didn't?
print(f"\n  Target column: '{target_col}'")
print("  Class distribution:")
counts = df[target_col].value_counts()
print(f"    0 (No Default) : {counts[0]} customers ({counts[0]/len(df)*100:.1f}%)")
print(f"    1 (Default)    : {counts[1]} customers ({counts[1]/len(df)*100:.1f}%)")
print("\n  ⚠️  Note: Dataset is IMBALANCED (fewer defaults than non-defaults)")
print("     We'll fix this using SMOTE technique later.")

# ── Plot 1: Class Distribution ──────────────────────────────
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
colors = ['#2ecc71', '#e74c3c']
df[target_col].value_counts().plot(kind='bar', color=colors, edgecolor='black')
plt.title('Class Distribution\n(0=No Default, 1=Default)', fontsize=12, fontweight='bold')
plt.xlabel('Default (0=No, 1=Yes)')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)

# ── Plot 2: Age Distribution ─────────────────────────────────
plt.subplot(2, 2, 2)
sns.histplot(df['AGE'], bins=30, kde=True, color='steelblue')
plt.title('Age Distribution of Customers', fontsize=12, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Count')

# ── Plot 3: Credit Limit by Default Status ───────────────────
plt.subplot(2, 2, 3)
df.groupby(target_col)['LIMIT_BAL'].mean().plot(kind='bar', color=colors, edgecolor='black')
plt.title('Avg Credit Limit: Default vs No Default', fontsize=12, fontweight='bold')
plt.xlabel('Default (0=No, 1=Yes)')
plt.ylabel('Average Credit Limit')
plt.xticks(rotation=0)

# ── Plot 4: Correlation Heatmap (top features) ───────────────
plt.subplot(2, 2, 4)
top_features = ['LIMIT_BAL', 'AGE', 'PAY_0', 'BILL_AMT1', 'PAY_AMT1', target_col]
corr = df[top_features].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Heatmap (Key Features)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=100, bbox_inches='tight')
plt.show()
print("\n  📈 EDA plots saved as 'eda_plots.png'")


# ────────────────────────────────────────────────────────────
# STEP 3: Data Preprocessing (Cleaning + Preparing the Data)
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[STEP 3] Preprocessing Data...")
print("=" * 60)

# 3a. Drop the 'ID' column - it's just a row number, not useful for prediction
if 'ID' in df.columns:
    df = df.drop('ID', axis=1)
    print("  ✅ Dropped 'ID' column (not useful for prediction)")

# 3b. Handle outliers in LIMIT_BAL (credit limit)
#     We'll clip extreme values to the 1st and 99th percentile
q1 = df['LIMIT_BAL'].quantile(0.01)
q99 = df['LIMIT_BAL'].quantile(0.99)
df['LIMIT_BAL'] = df['LIMIT_BAL'].clip(q1, q99)
print(f"  ✅ Clipped LIMIT_BAL outliers to range [{q1:.0f}, {q99:.0f}]")

# 3c. Fix invalid categorical values
#     SEX: should be 1 or 2 only
#     EDUCATION: should be 1-4 (1=grad, 2=university, 3=high school, 4=others)
#     MARRIAGE: should be 1-3
df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
print("  ✅ Fixed invalid values in EDUCATION and MARRIAGE columns")

# 3d. Separate Features (X) and Target (y)
#     X = everything we use to PREDICT (input)
#     y = what we want to PREDICT (output: default or not)
X = df.drop(target_col, axis=1)
y = df[target_col]
print(f"\n  Features (X): {X.shape[1]} columns")
print(f"  Target  (y): {y.shape[0]} rows")

# 3e. Split data into Training and Testing sets
#     80% of data → train the model
#     20% of data → test how well it learned
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% for testing
    random_state=42,     # fixed seed so results are reproducible
    stratify=y           # keep same class ratio in both splits
)
print(f"\n  Train set: {X_train.shape[0]} samples")
print(f"  Test  set: {X_test.shape[0]} samples")

# 3f. Handle Class Imbalance using SMOTE
#     SMOTE creates synthetic (artificial) examples of the minority class
#     so both classes have equal representation during training
print("\n  Applying SMOTE to fix class imbalance in training data...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"  Before SMOTE → Class 0: {sum(y_train==0)}, Class 1: {sum(y_train==1)}")
print(f"  After  SMOTE → Class 0: {sum(y_train_balanced==0)}, Class 1: {sum(y_train_balanced==1)}")

# 3g. Normalize / Scale the features
#     StandardScaler makes all columns have mean=0 and std=1
#     This helps models like Logistic Regression work much better
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)  # fit on train, then transform
X_test_scaled  = scaler.transform(X_test)                 # only transform test (no fitting!)
print("\n  ✅ Features scaled using StandardScaler")


# ────────────────────────────────────────────────────────────
# STEP 4: Train Machine Learning Models
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[STEP 4] Training Machine Learning Models...")
print("=" * 60)

# We'll train 3 models and compare them:
#   1. Logistic Regression  → simple, interpretable, great baseline
#   2. Random Forest        → ensemble of many decision trees
#   3. XGBoost              → state-of-the-art gradient boosting

# ── Model 1: Logistic Regression ─────────────────────────────
print("\n  Training Model 1: Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train_balanced)
print("  ✅ Logistic Regression trained!")

# ── Model 2: Random Forest ───────────────────────────────────
print("\n  Training Model 2: Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,   # 100 decision trees
    random_state=42,
    n_jobs=-1           # use all CPU cores for speed
)
rf_model.fit(X_train_scaled, y_train_balanced)
print("  ✅ Random Forest trained!")

# ── Model 3: XGBoost ─────────────────────────────────────────
print("\n  Training Model 3: XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)
xgb_model.fit(X_train_scaled, y_train_balanced)
print("  ✅ XGBoost trained!")


# ────────────────────────────────────────────────────────────
# STEP 5: Evaluate the Models
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[STEP 5] Evaluating Models on Test Data...")
print("=" * 60)

# A helper function to evaluate each model neatly
def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluates a trained model and prints key metrics.
    
    Metrics explained:
    - Precision : Of all predicted defaults, how many were actually defaults?
    - Recall    : Of all actual defaults, how many did we catch?
    - F1-Score  : Balance between Precision and Recall (higher = better)
    - ROC-AUC   : Overall ability to distinguish classes (1.0 = perfect)
    """
    # Make predictions
    y_pred      = model.predict(X_test)          # predicted class (0 or 1)
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # predicted probability

    # Calculate ROC-AUC score
    auc = roc_auc_score(y_test, y_pred_prob)

    print(f"\n  {'─'*40}")
    print(f"  📌 {model_name}")
    print(f"  {'─'*40}")
    print(f"  ROC-AUC Score: {auc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=['No Default', 'Default'],
                                 digits=4))
    return auc, y_pred, y_pred_prob

# Evaluate all 3 models
lr_auc,  lr_pred,  lr_prob  = evaluate_model(lr_model,  X_test_scaled, y_test, "Logistic Regression")
rf_auc,  rf_pred,  rf_prob  = evaluate_model(rf_model,  X_test_scaled, y_test, "Random Forest")
xgb_auc, xgb_pred, xgb_prob = evaluate_model(xgb_model, X_test_scaled, y_test, "XGBoost")


# ────────────────────────────────────────────────────────────
# STEP 6: Visualize Results
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[STEP 6] Generating Result Visualizations...")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Credit Card Default Prediction - Model Results', fontsize=14, fontweight='bold')

# ── Plot 1: ROC Curves for all 3 models ─────────────────────
ax1 = axes[0, 0]
for name, prob, auc in [
    ("Logistic Regression", lr_prob,  lr_auc),
    ("Random Forest",       rf_prob,  rf_auc),
    ("XGBoost",             xgb_prob, xgb_auc),
]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax1.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)

ax1.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curves - All Models')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# ── Plot 2: Model Comparison Bar Chart ──────────────────────
ax2 = axes[0, 1]
model_names = ['Logistic\nRegression', 'Random\nForest', 'XGBoost']
auc_scores  = [lr_auc, rf_auc, xgb_auc]
bar_colors  = ['#3498db', '#2ecc71', '#e74c3c']
bars = ax2.bar(model_names, auc_scores, color=bar_colors, edgecolor='black', width=0.5)
ax2.set_ylim(0.5, 1.0)
ax2.set_ylabel('ROC-AUC Score')
ax2.set_title('Model Comparison (ROC-AUC)')
ax2.grid(True, axis='y', alpha=0.3)
for bar, score in zip(bars, auc_scores):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

# ── Plot 3: Confusion Matrix for best model ──────────────────
# Find the best model based on AUC
best_idx  = np.argmax(auc_scores)
best_name = ['Logistic Regression', 'Random Forest', 'XGBoost'][best_idx]
best_pred = [lr_pred, rf_pred, xgb_pred][best_idx]

ax3 = axes[1, 0]
cm = confusion_matrix(y_test, best_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Default', 'Default'])
disp.plot(ax=ax3, colorbar=False, cmap='Blues')
ax3.set_title(f'Confusion Matrix\n(Best Model: {best_name})')

# ── Plot 4: Feature Importance (Random Forest) ───────────────
ax4 = axes[1, 1]
feature_names = X.columns.tolist()
importances   = rf_model.feature_importances_
# Show only top 10 most important features
top_n    = 10
indices  = np.argsort(importances)[::-1][:top_n]
top_feat = [feature_names[i] for i in indices]
top_imp  = importances[indices]

ax4.barh(range(top_n), top_imp[::-1], color='steelblue', edgecolor='black')
ax4.set_yticks(range(top_n))
ax4.set_yticklabels(top_feat[::-1])
ax4.set_xlabel('Importance Score')
ax4.set_title('Top 10 Feature Importances\n(Random Forest)')
ax4.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('model_results.png', dpi=100, bbox_inches='tight')
plt.show()
print("  📈 Model result plots saved as 'model_results.png'")


# ────────────────────────────────────────────────────────────
# STEP 7: Final Summary
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[STEP 7] Final Summary")
print("=" * 60)

print(f"""
  MODEL PERFORMANCE SUMMARY
  ┌──────────────────────────┬───────────┐
  │ Model                    │  ROC-AUC  │
  ├──────────────────────────┼───────────┤
  │ Logistic Regression      │  {lr_auc:.4f}  │
  │ Random Forest            │  {rf_auc:.4f}  │
  │ XGBoost                  │  {xgb_auc:.4f}  │
  └──────────────────────────┴───────────┘

  🏆 Best Model: {best_name} (AUC = {max(auc_scores):.4f})

  📌 Key Insights:
     • PAY_0 (repayment status last month) is the strongest predictor
     • Customers with higher credit limits tend to default less
     • SMOTE helped balance the class imbalance in training

  📂 Output files saved:
     • eda_plots.png     → Exploratory Data Analysis charts
     • model_results.png → Model comparison and evaluation charts

  🚀 Next Steps to improve your project:
     1. Try Neural Networks (using TensorFlow/Keras)
     2. Tune hyperparameters using GridSearchCV
     3. Add SHAP values to explain individual predictions
     4. Deploy the model using Flask or Streamlit
""")

print("=" * 60)
print("  ✅ Project Complete! Great work! 🎉")
print("=" * 60)
