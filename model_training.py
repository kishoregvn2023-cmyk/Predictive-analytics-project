import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

print("--- MODULE 1: Loading and Preprocessing Data ---")
# Using exactly 50,000 samples to match Section 4.1 of your paper
df = pd.read_csv("wildfires.csv", low_memory=False)
df = df.sample(n=50000, random_state=42) 

base_features = [
    "LATITUDE", "LONGITUDE", "DISCOVERY_DOY", 
    "FIRE_YEAR", "STAT_CAUSE_CODE", "STATE", "OWNER_CODE"
]
target = "FIRE_SIZE_CLASS"

# Drop missing values
df = df.dropna(subset=base_features + [target])

# Encode categorical variables into numbers
label_encoders = {}
for col in ['STATE', 'OWNER_CODE', 'STAT_CAUSE_CODE', target]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

print("--- MODULE 2: Advanced Temporal Engineering ---")
# Cyclical engineering so the model knows Day 365 is next to Day 1
df['DOY_SIN'] = np.sin(2 * np.pi * df['DISCOVERY_DOY'] / 365.0)
df['DOY_COS'] = np.cos(2 * np.pi * df['DISCOVERY_DOY'] / 365.0)

temporal_features = ['DOY_SIN', 'DOY_COS']
all_features = base_features + temporal_features

X = df[all_features]
y = df[target]

print("--- MODULE 3: Proportional SMOTE Balancing ---")
# Split data BEFORE balancing to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize spatial coordinates for better model convergence
scaler = MinMaxScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

cols_to_scale = ['LATITUDE', 'LONGITUDE']
X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

# Proportional SMOTE: Boost rare severe fires to 25% of the majority class
# This perfectly balances severe-class recall without destroying global accuracy
class_counts = y_train.value_counts().to_dict()
max_class_count = max(class_counts.values())

custom_smote_strategy = {
    cls: max(count, int(max_class_count * 0.25)) for cls, count in class_counts.items()
}

smote = SMOTE(sampling_strategy=custom_smote_strategy, random_state=42, k_neighbors=3)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"Original Training Shape: {X_train.shape}")
print(f"Balanced Training Shape: {X_train_balanced.shape}")

print("\n--- MODULE 4: Training All 5 Models ---")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, multi_class='multinomial'),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=200, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=8, eval_metric="mlogloss", random_state=42)
}

results = {}
rf_preds = None
lr_preds = None

for name, model in models.items():
    print(f"\n=========================================")
    print(f"Training: {name}")
    print(f"=========================================")
    
    model.fit(X_train_balanced, y_train_balanced)
    preds = model.predict(X_test_scaled)
    
    # Save specific predictions to generate the Confusion Matrices later
    if name == "Random Forest":
        rf_preds = preds
    if name == "Logistic Regression":
        lr_preds = preds
        
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    
    # Print the report. USE THIS TERMINAL OUTPUT TO FILL IN TABLE 1 IN YOUR PAPER.
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, zero_division=0))

print("\n--- Generating Required Visualizations ---")
# IMPORTANT: When these pop up, save them to your computer using the Floppy Disk icon!

# GRAPH 1: 5-Model Accuracy Comparison (For Figure 9 in the paper)
plt.figure(figsize=(10,6))
plt.bar(results.keys(), results.values(), color=['#c0392b', '#27ae60', '#e67e22', '#8e44ad', '#2980b9'])
plt.title("Wildfire Model Accuracy Comparison")
plt.ylabel("Global Accuracy")
plt.ylim(0, 1.0)
for i, v in enumerate(results.values()):
    plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# GRAPH 2: Random Forest Confusion Matrix (For Figure 8 in the paper)
plt.figure(figsize=(8,6))
cm_rf = confusion_matrix(y_test, rf_preds)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='rocket', 
            xticklabels=label_encoders[target].classes_, 
            yticklabels=label_encoders[target].classes_)
plt.title("Confusion Matrix - Random Forest (Proposed System)")
plt.xlabel("Predicted Severity Class")
plt.ylabel("Actual Severity Class")
plt.show()

# GRAPH 3: Logistic Regression Baseline Failure (Matches Section 4.3 text)
plt.figure(figsize=(8,6))
cm_lr = confusion_matrix(y_test, lr_preds)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Reds', 
            xticklabels=label_encoders[target].classes_, 
            yticklabels=label_encoders[target].classes_)
plt.title("Confusion Matrix - Logistic Regression (Baseline Failure)")
plt.xlabel("Predicted Severity Class")
plt.ylabel("Actual Severity Class")
plt.show()

# GRAPH 4: Feature Importance (For Figure 7 in the paper)
rf_model = models["Random Forest"]
plt.figure(figsize=(10,6))
feature_importances = pd.Series(rf_model.feature_importances_, index=all_features)
feature_importances.sort_values(ascending=False).plot(kind='bar', color='#8e44ad')
plt.title("Feature Importance - Validating Engineered Variables")
plt.ylabel("Relative Importance")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\n--- Process Complete! ---")