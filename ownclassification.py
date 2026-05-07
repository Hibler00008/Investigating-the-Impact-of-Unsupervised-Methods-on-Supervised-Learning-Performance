import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
from datetime import datetime
warnings.filterwarnings("ignore")

filename = "drybean.csv"
filepathstr = "datasets/"
# ================= OUTPUT REDIRECTION =================
# Create output directory if it doesn't exist
import os
os.makedirs('output', exist_ok=True)

# Generate timestamp for unique filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f'output/{filename}classification{timestamp}.txt'

# Redirect stdout to both file and console
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

# Start logging
logger = Logger(output_file)
sys.stdout = logger

print(f"Output will be saved to: {output_file}")
print(f"Timestamp: {timestamp}")
print("="*70)

from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import RandomizedSearchCV

# ================= DATA LOADING =================

filepath = filepathstr + filename

df = pd.read_csv(filepath)

print("="*50)
print("DATASET SHAPE:", df.shape)
print("="*50)

# ================= PREPROCESSING =================

def encode_last_column(df):
    """Encode target column if categorical"""
    last_col = df.columns[-1]
    if df[last_col].dtype == 'object' or not pd.api.types.is_numeric_dtype(df[last_col]):
        le = LabelEncoder()
        df[last_col] = le.fit_transform(df[last_col])
    return df

def split_data(df, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def remove_mahalanobis_outliers(X_train, y_train, iqr_multiplier=1.5):
    """Remove outliers using Mahalanobis distance"""
    mat = X_train.to_numpy()
    mean_vec = mat.mean(axis=0)
    cov = np.cov(mat, rowvar=False)
    cov_inv = np.linalg.inv(cov)
    dists = np.array([mahalanobis(x, mean_vec, cov_inv) for x in mat])
    q1, q3 = np.quantile(dists, [0.25, 0.75])
    iqr = q3 - q1
    cutoff = q3 + iqr_multiplier * iqr
    mask = dists <= cutoff
    removed = np.sum(~mask)
    X_train.drop(X_train.index[~mask], inplace=True)
    y_train.drop(y_train.index[~mask], inplace=True)
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    print(f"Removed {removed} outliers out of {len(dists)} samples.")
    return removed

def normalize_train_test(X_train, X_test):
    mean = X_train.mean()
    std = X_train.std(ddof=0)
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    return X_train_scaled, X_test_scaled

def get_class_weights(y_train):
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    print("Computed class weights:", class_weight_dict)
    return class_weight_dict

df = encode_last_column(df)

X_train, X_test, Y_train, Y_test = split_data(df, test_size=0.2, random_state=42)

removed_outliers = remove_mahalanobis_outliers(X_train, Y_train, iqr_multiplier=1.5)

X_train_scaled, X_test_scaled = normalize_train_test(X_train, X_test)

class_weight_dict = get_class_weights(Y_train)

print("\n✅ Preprocessing Complete!")
print(f"Training shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")

print("\n" + "="*50)
print("TRAINING BASE MODELS")
print("="*50)

base_models = {
    "Logistic Regression": LogisticRegression(class_weight=class_weight_dict, max_iter=5000),
    "Decision Tree": DecisionTreeClassifier(class_weight=class_weight_dict, random_state=42),
    "Random Forest": RandomForestClassifier(class_weight=class_weight_dict, random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

base_results = {}

for name, model in base_models.items():
    model.fit(X_train_scaled, Y_train)
    Y_test_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(Y_test, Y_test_pred)
    test_f1 = f1_score(Y_test, Y_test_pred, average='weighted')
    base_results[name] = test_accuracy
    
    print(f"\n{name}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")

# ================= HYPERPARAMETER TUNING =================

print("\n" + "="*50)
print("HYPERPARAMETER TUNING WITH RANDOMIZED SEARCH")
print("="*50)

param_distributions = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
        "penalty": ['l2', 'l1', 'elasticnet', 'none'],
        "solver": ['saga'],
        "l1_ratio": [0, 0.25, 0.5, 0.75, 1.0],
        "max_iter": [1000, 2000, 5000]
    },
    "Decision Tree": {
        "min_samples_split": [20, 50, 100, 200, 300],
        "criterion": ["gini", "entropy", "log_loss"],
        "max_features": ['sqrt', 'log2', 0.5, 0.7, None],
        "min_impurity_decrease": [0.0, 0.001, 0.005, 0.01],
        "ccp_alpha": [0.0, 0.001, 0.01, 0.05]
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300, 400, 500],
        "min_samples_split": [20, 50, 100, 200],
        "min_samples_leaf": [10, 20, 50, 100],
        "max_features": ['sqrt', 'log2', 0.5, 0.7],
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 150, 200, 300, 400],
        "learning_rate": [0.01, 0.05, 0.1, 0.5, 0.8, 1.0, 1.2],
        "algorithm": ["SAMME", "SAMME.R"],
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200, 300, 400, 500],
        "learning_rate": [0.005, 0.01, 0.05, 0.1, 0.15, 0.2],
        "min_samples_split": [50, 100, 200, 300],
        "min_samples_leaf": [20, 50, 100, 150],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "max_features": ['sqrt', 'log2', 0.5, 0.7, None],
        "n_iter_no_change": [5, 10, 15]
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300, 400, 500],
        "learning_rate": [0.005, 0.01, 0.05, 0.1, 0.15, 0.2],
        "min_child_weight": [1, 3, 5, 7, 10],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bylevel": [0.6, 0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.01, 0.1, 0.2, 0.5],
        "reg_alpha": [0, 0.001, 0.01, 0.1, 0.5, 1.0],
        "reg_lambda": [0.1, 0.5, 1.0, 2.0, 5.0],
        "scale_pos_weight": [1, 2, 5, 10]
    }
}

best_params = {}

for name in base_models.keys():
    print(f"\nTuning {name}...")
    model = base_models[name]
    
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions[name],
        n_iter=50,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    random_search.fit(X_train_scaled, Y_train.values.ravel())
    best_params[name] = random_search.best_params_
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best CV score: {random_search.best_score_:.4f}")

# ================= FINAL TRAINING WITH BEST PARAMETERS =================

print("\n" + "="*50)
print("FINAL MODELS WITH TUNED PARAMETERS")
print("="*50)

final_models = {
    "Logistic Regression": LogisticRegression(
        class_weight=class_weight_dict,
        max_iter=5000,
        **best_params["Logistic Regression"]
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight=class_weight_dict,
        random_state=42,
        **best_params["Decision Tree"]
    ),
    "Random Forest": RandomForestClassifier(
        class_weight=class_weight_dict,
        random_state=42,
        **best_params["Random Forest"]
    ),
    "AdaBoost": AdaBoostClassifier(
        random_state=42,
        **best_params["AdaBoost"]
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42,
        **best_params["Gradient Boosting"]
    ),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        **best_params["XGBoost"]
    )
}

final_results = {}
feature_names = X_train.columns.tolist()

for name, model in final_models.items():
    print(f"\n{'='*50}")
    print(f"{name}")
    print('='*50)
    
    model.fit(X_train_scaled, Y_train)
    
    Y_train_pred = model.predict(X_train_scaled)
    Y_test_pred = model.predict(X_test_scaled)
    
    train_accuracy = accuracy_score(Y_train, Y_train_pred)
    train_f1 = f1_score(Y_train, Y_train_pred, average='weighted')
    train_precision = precision_score(Y_train, Y_train_pred, average='weighted')
    train_recall = recall_score(Y_train, Y_train_pred, average='weighted')
    
    test_accuracy = accuracy_score(Y_test, Y_test_pred)
    test_f1 = f1_score(Y_test, Y_test_pred, average='weighted')
    test_precision = precision_score(Y_test, Y_test_pred, average='weighted')
    test_recall = recall_score(Y_test, Y_test_pred, average='weighted')
    
    final_results[name] = test_accuracy
    
    print("\nModel Performance for Training Set:")
    print(f"  Accuracy:  {train_accuracy:.4f}")
    print(f"  F1 Score:  {train_f1:.4f}")
    print(f"  Precision: {train_precision:.4f}")
    print(f"  Recall:    {train_recall:.4f}")
    
    print("\nModel Performance for Test Set:")
    print(f"  Accuracy:  {test_accuracy:.4f}")
    print(f"  F1 Score:  {test_f1:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    
    # Feature Importance (for all models)
    if hasattr(model, 'feature_importances_'):
        # Tree-based models: use feature_importances_
        importances = model.feature_importances_
        importance_type = "Feature Importance"
    elif hasattr(model, 'coef_'):
        # Linear models: use absolute values of coefficients
        if len(model.coef_.shape) > 1:
            # Multi-class: average absolute coefficients across classes
            importances = np.abs(model.coef_).mean(axis=0)
        else:
            # Binary classification
            importances = np.abs(model.coef_[0])
        importance_type = "Coefficient Magnitude"
    else:
        # Models without interpretable feature importance (e.g., KNN)
        print("\n⚠️ This model does not provide feature importance.")
        continue
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features ({importance_type}):")
    print(feature_importance.head(10).to_string(index=False))
    
    # Plot feature importance (sorted in descending order)
    plt.figure(figsize=(10, 6))
    top_features = feature_importance.head(10).iloc[::-1]  # Reverse for ascending plot
    sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
    plt.title(f'Top 10 Feature Importances - {name}')
    plt.xlabel(importance_type)
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

# ================= FINAL COMPARISON PLOTS =================

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Base models comparison
axes[0].barh(list(base_results.keys()), list(base_results.values()), color='skyblue')
axes[0].set_xlabel('Test Accuracy')
axes[0].set_title('Base Models Performance')
axes[0].set_xlim([0, 1])

# Tuned models comparison
axes[1].barh(list(final_results.keys()), list(final_results.values()), color='coral')
axes[1].set_xlabel('Test Accuracy')
axes[1].set_title('Tuned Models Performance')
axes[1].set_xlim([0, 1])

plt.tight_layout()
plt.show()

# Print final comparison
print("\n" + "="*50)
print("FINAL ACCURACY COMPARISON")
print("="*50)
comparison_df = pd.DataFrame({
    'Model': list(final_results.keys()),
    'Base Accuracy': [base_results[m] for m in final_results.keys()],
    'Tuned Accuracy': list(final_results.values()),
    'Improvement': [final_results[m] - base_results[m] for m in final_results.keys()]
})
print(comparison_df.to_string(index=False))

print("\n✅ Training Complete!")
print(f"Best Model: {max(final_results, key=final_results.get)}")
print(f"Best Accuracy: {max(final_results.values()):.4f}")

# ================= CLOSE LOGGER =================
print(f"\n{'='*70}")
print(f"Results saved to: {output_file}")
logger.close()
sys.stdout = logger.terminal  # Restore original stdout