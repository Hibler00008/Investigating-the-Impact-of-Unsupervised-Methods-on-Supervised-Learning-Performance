import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import os
import xgboost as xgb
xgb_logger = xgb.callback.TrainingCallback()
os.environ['XGBOOST_VERBOSITY'] = '0'
warnings.filterwarnings("ignore")
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  

output_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(output_dir, exist_ok=True)

DATASET_PATH = os.path.join("datasets", "16P.csv")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
dataset_name = os.path.splitext(os.path.basename(DATASET_PATH))[0]
output_file = os.path.join(output_dir, f'{dataset_name}_results_{timestamp}.txt')

class ThreadSafeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8', buffering=1)  
        self.lock = threading.Lock()
        self.buffer = {}  
    
    def write(self, message):
        with self.lock:
            self.terminal.write(message)
            self.log.write(message)
            self.terminal.flush()
            self.log.flush()
    
    def flush(self):
        with self.lock:
            self.terminal.flush()
            self.log.flush()
    
    def buffer_write(self, thread_id, message):
        if thread_id not in self.buffer:
            self.buffer[thread_id] = []
        self.buffer[thread_id].append(message)
    
    def flush_buffer(self, thread_id):
        if thread_id in self.buffer:
            with self.lock:
                output = ''.join(self.buffer[thread_id])
                self.terminal.write(output)
                self.log.write(output)
                self.terminal.flush()
                self.log.flush()
                del self.buffer[thread_id]
    
    def close(self):
        with self.lock:
            self.log.close()

logger = ThreadSafeLogger(output_file)
sys.stdout = logger

print(f"Output will be saved to: {output_file}")
print(f"Timestamp: {timestamp}")
print("="*70)

from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def encode_last_column(df):
    last_col = df.columns[-1]
    if df[last_col].dtype == 'object' or not pd.api.types.is_numeric_dtype(df[last_col]):
        le = LabelEncoder()
        df[last_col] = le.fit_transform(df[last_col])
    return df

def split_data(df, test_size=0.2, random_state=42):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def remove_mahalanobis_outliers(X_train, y_train, iqr_multiplier=1.5):
   
    if X_train.shape[1] == 1:
        
        col = X_train.columns[0]
        values = X_train[col].to_numpy()
        mean = values.mean()
        std = values.std()

        if std == 0:
            return 0

        z_scores = np.abs((values - mean) / std)
        mask = z_scores < 3  # keep values with |z| < 3

        removed = np.sum(~mask)

        X_train.drop(X_train.index[~mask], inplace=True)
        y_train.drop(y_train.index[~mask], inplace=True)
        X_train.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)

        return removed

    mat = X_train.to_numpy()
    mean_vec = mat.mean(axis=0)
    cov = np.cov(mat, rowvar=False)

    cov += np.eye(cov.shape[0]) * 1e-9

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

    print(f"Removed {removed} outliers using Mahalanobis distance.")
    return removed


def normalize_train_test(X_train, X_test):
    mean = X_train.mean()
    std = X_train.std(ddof=0)
    std = std.replace(0, 1) 
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    return X_train_scaled, X_test_scaled

def get_class_weights(y_train):
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    print("Computed class weights:", class_weight_dict)
    return class_weight_dict


class BufferedOutput:
    """Context manager for buffered output in threads"""
    def __init__(self, logger, thread_id):
        self.logger = logger
        self.thread_id = thread_id
        self.output_buffer = []
        
    def __enter__(self):
        self.old_stdout = sys.stdout
        sys.stdout = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout
        # Flush all buffered output at once
        self.logger.flush_buffer(self.thread_id)
    
    def write(self, message):
        self.logger.buffer_write(self.thread_id, message)
    
    def flush(self):
        pass


def evaluate_model_metrics(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    return {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'train_f1': f1_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'train_precision': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'train_recall': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_f1': f1_score(y_test, y_test_pred, average='weighted', zero_division=0),
        'test_precision': precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
        'test_recall': recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    }

def get_feature_importance_data(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        importance_type = "Feature Importance"
    elif hasattr(model, 'coef_'):
        if len(model.coef_.shape) > 1:
            importances = np.abs(model.coef_).mean(axis=0)
        else:
            importances = np.abs(model.coef_[0])
        importance_type = "Coefficient Magnitude"
    else:
        return None, None
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return feature_importance, importance_type

def train_model_baseline(model_config):
    model_name = model_config['name']
    model_class = model_config['class']
    model_params = model_config['params']
    X_train = model_config['X_train']
    y_train = model_config['y_train']
    X_test = model_config['X_test']
    y_test = model_config['y_test']
    
    thread_id = threading.current_thread().ident
    
    with BufferedOutput(logger, thread_id):
        print(f"\n{'#'*70}")
        print(f"# {model_name} - BASELINE")
        print(f"{'#'*70}")
        print(f"Training baseline model...")
        
        start_time = time.time()
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        metrics = evaluate_model_metrics(model, X_train, y_train, X_test, y_test)
        
        print(f"\nTraining completed in {train_time:.2f} seconds")
        print(f"\nTraining Set Performance:")
        print(f"  Accuracy:  {metrics['train_accuracy']:.4f}")
        print(f"  F1 Score:  {metrics['train_f1']:.4f}")
        print(f"  Precision: {metrics['train_precision']:.4f}")
        print(f"  Recall:    {metrics['train_recall']:.4f}")
        
        print(f"\nTest Set Performance:")
        print(f"  Accuracy:  {metrics['test_accuracy']:.4f}")
        print(f"  F1 Score:  {metrics['test_f1']:.4f}")
        print(f"  Precision: {metrics['test_precision']:.4f}")
        print(f"  Recall:    {metrics['test_recall']:.4f}")
    
    return {
        'name': model_name,
        'model': model,
        'metrics': metrics,
        'train_time': train_time
    }

def tune_model_hyperparameters(model_config):
    model_name = model_config['name']
    model_class = model_config['class']
    model_params = model_config['params']
    param_distributions = model_config['param_grid']
    X_train = model_config['X_train']
    y_train = model_config['y_train']
    X_test = model_config['X_test']
    y_test = model_config['y_test']
    n_iter = model_config.get('n_iter', 50)  
    
    thread_id = threading.current_thread().ident
    
    with BufferedOutput(logger, thread_id):
        print(f"\n{'#'*70}")
        print(f"# {model_name} - HYPERPARAMETER TUNING (RANDOMIZED)")
        print(f"{'#'*70}")
        print(f"Tuning hyperparameters with RandomizedSearchCV ({n_iter} iterations)...")
        print(f"Starting at: {datetime.now().strftime('%H:%M:%S')}")
        
        start_time = time.time()
        base_model = model_class(**model_params)
        random_search = RandomizedSearchCV(
            base_model, param_distributions, 
            n_iter=n_iter,
            cv=3,
            scoring="accuracy", 
            n_jobs=1,  
            verbose=2, 
            random_state=42
        )
        random_search.fit(X_train, y_train.values.ravel())
        model = random_search.best_estimator_
        tune_time = time.time() - start_time
        
        metrics = evaluate_model_metrics(model, X_train, y_train, X_test, y_test)
        
        print(f"\nTuning completed in {tune_time:.2f} seconds")
        print(f"Finished at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"Best CV score: {random_search.best_score_:.4f}")
        print(f"\nBest parameters:")
        for param, value in random_search.best_params_.items():
            print(f"  {param}: {value}")
        
        print(f"\nTraining Set Performance:")
        print(f"  Accuracy:  {metrics['train_accuracy']:.4f}")
        print(f"  F1 Score:  {metrics['train_f1']:.4f}")
        print(f"  Precision: {metrics['train_precision']:.4f}")
        print(f"  Recall:    {metrics['train_recall']:.4f}")
        
        print(f"\nTest Set Performance:")
        print(f"  Accuracy:  {metrics['test_accuracy']:.4f}")
        print(f"  F1 Score:  {metrics['test_f1']:.4f}")
        print(f"  Precision: {metrics['test_precision']:.4f}")
        print(f"  Recall:    {metrics['test_recall']:.4f}")
    
    return {
        'name': model_name,
        'model': model,
        'metrics': metrics,
        'best_params': random_search.best_params_,
        'cv_score': random_search.best_score_,
        'tune_time': tune_time
    }

def display_feature_importance(model_result, feature_names):
    """Display feature importance (buffered output)"""
    model_name = model_result['name']
    model = model_result['model']
    
    thread_id = threading.current_thread().ident
    
    with BufferedOutput(logger, thread_id):
        print(f"\n{'#'*70}")
        print(f"# {model_name} - FEATURE IMPORTANCE")
        print(f"{'#'*70}")
        
        feature_importance, importance_type = get_feature_importance_data(model, feature_names)
        
        if feature_importance is None:
            print("\n⚠️ This model does not provide feature importance.")
            return None
        
        print(f"\nTop 10 Most Important Features ({importance_type}):")
        print(feature_importance.head(10).to_string(index=False))
    
    return feature_importance, importance_type


df = pd.read_csv(DATASET_PATH)
print("\nDataset Shape:", df.shape)

df = encode_last_column(df)
X_train, X_test, Y_train, Y_test = split_data(df, test_size=0.2, random_state=42)
removed_outliers = remove_mahalanobis_outliers(X_train, Y_train, iqr_multiplier=1.5)
X_train_scaled, X_test_scaled = normalize_train_test(X_train, X_test)
class_weight_dict = get_class_weights(Y_train)

feature_names = X_train.columns.tolist()


model_configs = {
    "Logistic Regression": {
        'name': "Logistic Regression",
        'class': LogisticRegression,
        'params': {'class_weight': class_weight_dict, 'max_iter': 5000, 'random_state': 42},
        'param_grid': {
            "C": [0.1, 1, 10, 100],
            "penalty": ['l2', 'l1'],
            "solver": ['saga']
        },
        'X_train': X_train_scaled,
        'y_train': Y_train,
        'X_test': X_test_scaled,
        'y_test': Y_test
    },
    "Decision Tree": {
        'name': "Decision Tree",
        'class': DecisionTreeClassifier,
        'params': {'class_weight': class_weight_dict, 'random_state': 42},
        'param_grid': {
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [50, 100, 200],
            "min_samples_leaf": [20, 50, 100],
            "criterion": ["gini", "entropy"]
        },
        'X_train': X_train_scaled,
        'y_train': Y_train,
        'X_test': X_test_scaled,
        'y_test': Y_test
    },
    "Random Forest": {
        'name': "Random Forest",
        'class': RandomForestClassifier,
        'params': {'class_weight': class_weight_dict, 'random_state': 42, 'n_jobs': 1},
        'param_grid': {
            "n_estimators": [100, 200, 300],
            "max_depth": [20, 30, None],
            "min_samples_split": [50, 100, 200],
            "min_samples_leaf": [20, 50, 100],
            "max_features": ['sqrt', 'log2']
        },
        'X_train': X_train_scaled,
        'y_train': Y_train,
        'X_test': X_test_scaled,
        'y_test': Y_test
    },
    "AdaBoost": {
        'name': "AdaBoost",
        'class': AdaBoostClassifier,
        'params': {'random_state': 42},
        'param_grid': {
            "n_estimators": [50, 100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.5, 1.0]
        },
        'X_train': X_train_scaled,
        'y_train': Y_train,
        'X_test': X_test_scaled,
        'y_test': Y_test
    },
    "Gradient Boosting": {
        'name': "Gradient Boosting",
        'class': GradientBoostingClassifier,
        'params': {'random_state': 42},
        'param_grid': {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 1.0],
            "min_samples_split": [100, 200]
        },
        'X_train': X_train_scaled,
        'y_train': Y_train,
        'X_test': X_test_scaled,
        'y_test': Y_test
    },
    "XGBoost": {
        'name': "XGBoost",
        'class': XGBClassifier,
        'params': {'use_label_encoder': False, 'eval_metric': 'mlogloss', 'random_state': 42, 'n_jobs': 1,'verbosity': 0},
        'param_grid': {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "gamma": [0, 0.1, 0.2],
            "reg_alpha": [0, 0.1, 1.0],
            "reg_lambda": [1, 2, 5]
        },
        'X_train': X_train_scaled,
        'y_train': Y_train,
        'X_test': X_test_scaled,
        'y_test': Y_test
    }
}


print("\n" + "="*70)
print("PHASE 1: TRAINING BASELINE MODELS (PARALLEL)")
print("="*70)

baseline_results = {}
with ThreadPoolExecutor(max_workers=3) as executor:  # Reduced from 6 to 3
    futures = {executor.submit(train_model_baseline, config): name 
               for name, config in model_configs.items()}
    
    completed = 0
    total = len(futures)
    for future in as_completed(futures):
        result = future.result()
        baseline_results[result['name']] = result
        completed += 1
        print(f"\n[Progress: {completed}/{total}] Completed: {result['name']}")
        print("="*70)

# Plot baseline comparison
print("\n" + "="*70)
print("BASELINE MODEL COMPARISON")
print("="*70)

baseline_comparison = pd.DataFrame({
    'Model': [r['name'] for r in baseline_results.values()],
    'Test Accuracy': [r['metrics']['test_accuracy'] for r in baseline_results.values()],
    'Test F1': [r['metrics']['test_f1'] for r in baseline_results.values()],
    'Train Time (s)': [r['train_time'] for r in baseline_results.values()]
})
print(baseline_comparison.to_string(index=False))

plt.figure(figsize=(12, 6))
plt.barh(baseline_comparison['Model'], baseline_comparison['Test Accuracy'], color='skyblue')
plt.xlabel('Test Accuracy')
plt.title('Baseline Model Performance')
plt.xlim([0, 1])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{dataset_name}_baseline_{timestamp}.png'), dpi=150, bbox_inches='tight')
plt.close()

# ================= PHASE 2: HYPERPARAMETER TUNING (PARALLEL) =================

print("\n" + "="*70)
print("PHASE 2: HYPERPARAMETER TUNING WITH RANDOMIZED SEARCH (PARALLEL)")
print("="*70)

tuned_results = {}
with ThreadPoolExecutor(max_workers=3) as executor:  # Reduced from 6 to 3
    futures = {executor.submit(tune_model_hyperparameters, config): name 
               for name, config in model_configs.items()}
    
    completed = 0
    total = len(futures)
    for future in as_completed(futures):
        result = future.result()
        tuned_results[result['name']] = result
        completed += 1
        print(f"\n[Progress: {completed}/{total}] Completed tuning: {result['name']}")
        print("="*70)

# Plot tuned comparison
print("\n" + "="*70)
print("TUNED MODEL COMPARISON")
print("="*70)

tuned_comparison = pd.DataFrame({
    'Model': [r['name'] for r in tuned_results.values()],
    'Test Accuracy': [r['metrics']['test_accuracy'] for r in tuned_results.values()],
    'Test F1': [r['metrics']['test_f1'] for r in tuned_results.values()],
    'CV Score': [r['cv_score'] for r in tuned_results.values()],
    'Tune Time (s)': [r['tune_time'] for r in tuned_results.values()]
})
print(tuned_comparison.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Baseline vs Tuned
comparison_data = pd.DataFrame({
    'Model': baseline_comparison['Model'],
    'Baseline': baseline_comparison['Test Accuracy'],
    'Tuned': tuned_comparison['Test Accuracy']
})

x = np.arange(len(comparison_data))
width = 0.35
axes[0].barh(x - width/2, comparison_data['Baseline'], width, label='Baseline', color='skyblue')
axes[0].barh(x + width/2, comparison_data['Tuned'], width, label='Tuned', color='coral')
axes[0].set_yticks(x)
axes[0].set_yticklabels(comparison_data['Model'])
axes[0].set_xlabel('Test Accuracy')
axes[0].set_title('Baseline vs Tuned Models (RandomizedSearch)')
axes[0].legend()
axes[0].set_xlim([0, 1])

# Final metrics
metrics_data = tuned_comparison.set_index('Model')[['Test Accuracy', 'Test F1', 'CV Score']].T
metrics_data.plot(kind='bar', ax=axes[1])
axes[1].set_title('Tuned Model Metrics')
axes[1].set_ylabel('Score')
axes[1].set_ylim([0, 1])
axes[1].legend(loc='lower right')
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{dataset_name}_comparison_{timestamp}.png'), dpi=150, bbox_inches='tight')
plt.close()

# ================= PHASE 3: FEATURE IMPORTANCE (PARALLEL) =================

print("\n" + "="*70)
print("PHASE 3: FEATURE IMPORTANCE ANALYSIS (PARALLEL)")
print("="*70)

feature_importance_data = {}
with ThreadPoolExecutor(max_workers=6) as executor:
    futures = {executor.submit(display_feature_importance, result, feature_names): result['name'] 
               for result in tuned_results.values()}
    
    for future in as_completed(futures):
        model_name = futures[future]
        result = future.result()
        if result is not None:
            feature_importance_df, importance_type = result
            feature_importance_data[model_name] = (feature_importance_df, importance_type)

# Plot feature importance
for model_name, (feature_importance_df, importance_type) in feature_importance_data.items():
    plt.figure(figsize=(10, 6))
    top_features = feature_importance_df.head(10).iloc[::-1]
    sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
    plt.title(f'Top 10 Feature Importances - {model_name}')
    plt.xlabel(importance_type)
    plt.ylabel('Feature')
    plt.tight_layout()
    safe_model_name = model_name.replace(' ', '_')
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_{safe_model_name}_importance_{timestamp}.png'), dpi=150, bbox_inches='tight')
    plt.close()

# ================= FINAL SUMMARY =================

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

# Best model
best_model_name = max(tuned_results, key=lambda x: tuned_results[x]['metrics']['test_accuracy'])
best_accuracy = tuned_results[best_model_name]['metrics']['test_accuracy']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Test Accuracy: {best_accuracy:.4f}")
print(f"   Test F1 Score: {tuned_results[best_model_name]['metrics']['test_f1']:.4f}")

# Improvement summary
print(f"\n📈 Improvement Summary:")
for model_name in baseline_results.keys():
    baseline_acc = baseline_results[model_name]['metrics']['test_accuracy']
    tuned_acc = tuned_results[model_name]['metrics']['test_accuracy']
    improvement = (tuned_acc - baseline_acc) * 100
    print(f"   {model_name}: {improvement:+.2f}%")

# Time comparison
print(f"\n⏱️ Time Savings with RandomizedSearchCV:")
print("   Note: RandomizedSearch explores a subset of the parameter space,")
print("   providing faster results while still finding good hyperparameters.")

print("\n✅ All Training Complete!")

# ================= CLOSE LOGGER =================
print(f"\n{'='*70}")
print(f"Results saved to: {output_file}")
print(f"Plots saved to: {output_dir}")
sys.stdout = logger.terminal
logger.close()