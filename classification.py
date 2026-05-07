import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_auc_score,roc_curve, auc
from sklearn.preprocessing import label_binarize

from sklearn.decomposition import PCA

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


df = pd.read_excel("16P.csv")

def EDA(df):
    print("First 10 Observations Of Dataset\n")
    print(df.head(10))
    print("\n")
    print("Sample Observations Of Dataset\n")
    print(df.sample(10))
    print("\n")
    print("Features Of Dataset\n")
    print(df.columns)
    print("\n")
    print("Shape Of Dataset\n")
    print(df.shape)
    print("\n")
    print("Size Of Dataset\n")
    print(df.size)
    print("\n")
    print("Data Description\n")
    print(df.describe())
    print("\n")
    print("Data Info\n")
    print(df.info())
    print("\n")
    print("Total Null Values in Whole Dataset\n")
    print(df.isnull().sum().sum())
    print("\n")
    print("Total Null Values In Each Feature\n")
    print(df.isnull().sum())
    print("\n")
    print("Value Counts in Each Features\n")
    columns = np.array(df.columns)
    for column in columns:
        print(df[column].value_counts())
        print("\n")
    num_features = [feature for feature in df.columns if df[feature].dtype != 'O']
    print('Num of Numerical Features :', len(num_features))
    print("Numerical Features: ", num_features)
    cat_features = [feature for feature in df.columns if df[feature].dtype == 'O']
    print('\nNum of Categorical Features :', len(cat_features))
    print("Categorical Features: ", cat_features)
    discrete_features=[feature for feature in num_features if len(df[feature].unique())<=25]
    print('\nNum of Discrete Features :',len(discrete_features))
    print("Discrete Features: ", discrete_features)
    continuous_features=[feature for feature in num_features if feature not in discrete_features]
    print('\nNum of Continuous Features :',len(continuous_features))
    print("Continous Features: ", continuous_features)

    df.drop(columns=['Class']).hist(figsize=(10,8), bins=20)
    plt.suptitle("Feature Histograms")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,10))
    for i, col in enumerate(df.drop(columns=['Class']).columns, 1):
        plt.subplot(4, 4, i)
        sns.boxplot(data=df, x='Class', y=col)
        plt.title(f"Boxplot of {col} by Class")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,6))
    corr = df.drop(columns=['Class']).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()


EDA(df=df)


outlier_df = df.copy()

def Outliers(df):
    
    results = {}

    data = df.iloc[:, 1:9].to_numpy()
    print("data shape:", data.shape)

    if len(data) > 1:
        means = []
        for j in range(data.shape[1]):
            col_sum = 0
            for i in range(data.shape[0]):
                col_sum += data[i][j]
            means.append(col_sum / data.shape[0])
        means = np.array(means)
        print("means shape:", means.shape)
       
        centered = []
        for i in range(data.shape[0]):
            row = []
            for j in range(data.shape[1]):
                row.append(data[i][j] - means[j])
            centered.append(row)
        centered = np.array(centered)
        print("centered shape:", centered.shape)
        
        n = data.shape[0]
        sigma = (centered.T @ centered) / (n - 1)
        print("sigma shape:", sigma.shape)

        sigmainv = np.linalg.inv(sigma)
        print("sigmainv shape:", sigmainv.shape)

        mahalanobisdist = []
        for i in range(data.shape[0]):
            diff = centered[i]
            d = np.sqrt(diff @ sigmainv @ diff.T)
            mahalanobisdist.append(d)
        df['Mahalanobis'] = np.array(mahalanobisdist)
        mahalanobis = np.array(mahalanobisdist)
        print("mahalanobisdist shape:", mahalanobis.shape)


        md_q1, md_q3 = np.percentile(mahalanobisdist, [25, 75])
        md_iqr = md_q3 - md_q1
        threshold = md_q3 + 1.5 * md_iqr
        results["Mahalanobis"] = df[df["Mahalanobis"] > threshold]

    else:
        results["Mahalanobis"] = pd.DataFrame()
    mahalanobis_outliers = len(results["Mahalanobis"])
    print("Mahalanobis Outliers: %s" % mahalanobis_outliers)
    print("Outlier DF:\n", df.head())
    
    # OUTLIERS IN CSV FILES
    # results["Mahalanobis"].to_csv('_mahalanobis_outliers.csv')

    return results


Outliers(df=outlier_df)


## OUTLIER REMOVAL THROUGH MAHALANOBIS DISTANCE
maha = outlier_df['Mahalanobis']
df['Mahalanobis'] = maha
print(df.head())

md_q1, md_q3 = np.percentile(maha, [25, 75])
md_iqr = md_q3 - md_q1
threshold = md_q3 + 1.5 * md_iqr
df = df[df["Mahalanobis"] <= threshold]
print("\n Shape of Dataset after Outlier Removal")
print(df.shape)
print("\n")

df.drop('Mahalanobis', axis=1, inplace=True)
print(df.head())


X = df.drop('Class', axis=1)

label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(df['Class'])
Y = pd.DataFrame(Y_encoded)

## STRATIFIED TRAIN TEST SPLIT (ENSURES CLASS BALANCE IN BOTH SETS)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y,random_state=42)
print("Training Set Shape:", X_train.shape, Y_train.shape)
print("Test Set Shape:", X_test.shape, Y_test.shape)
print("\nClass Distribution in Training Set:\n", Y_train.value_counts(normalize=True))
print("\nClass Distribution in Test Set:\n", Y_test.value_counts(normalize=True))


## FEATURE SCALING (ROBUST SCALING)
scaler = RobustScaler()

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("\nScaled Training Data (first 5 rows):\n", X_train_scaled.head())


## HANDLING CLASS IMBALANCE
# Example: Class weights (to be used in model training later)
# Some models like LogisticRegression, RandomForest, XGBoost support "class_weight"

print("\nClass Distribution in Training Set (Counts):\n", Counter(Y_train))

classes = np.unique(Y_train)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=Y_train)
class_weight_dict = dict(zip(classes, class_weights))

print("\nComputed Class Weights:", class_weight_dict)




## MODEL TRAINING

models={
    "Logisitic Regression": LogisticRegression(class_weight=class_weight_dict),
    "K-Nearest Neighbors": KNeighborsClassifier(),  # No class_weight support
    "Support Vector Machine": SVC(class_weight=class_weight_dict, probability=True),
    "Decision Tree": DecisionTreeClassifier(class_weight=class_weight_dict),
    "Random Forest": RandomForestClassifier(class_weight=class_weight_dict),
    "AdaBoost": AdaBoostClassifier(),  # Weak learners ignore class_weight
    "Gradient Boosting": GradientBoostingClassifier(),  # Not supported
    "XGBoost": XGBClassifier()  # Use scale_pos_weight separately if binary
}


results = {}
for name, model in models.items():

    model.fit(X_train_scaled, Y_train) 

    Y_train_pred = model.predict(X_train_scaled)
    Y_test_pred = model.predict(X_test_scaled)

    model_train_accuracy = accuracy_score(Y_train, Y_train_pred) 
    model_train_f1 = f1_score(Y_train, Y_train_pred, average='weighted') 
    model_train_precision = precision_score(Y_train, Y_train_pred, average='weighted') 
    model_train_recall = recall_score(Y_train, Y_train_pred, average='weighted') 

    model_test_accuracy = accuracy_score(Y_test, Y_test_pred) 
    model_test_f1 = f1_score(Y_test, Y_test_pred, average='weighted') 
    model_test_precision = precision_score(Y_test, Y_test_pred, average='weighted') 
    model_test_recall = recall_score(Y_test, Y_test_pred, average='weighted') 

    if hasattr(model, "predict_proba"):
        y_train_scores = model.predict_proba(X_train_scaled)
        y_test_scores = model.predict_proba(X_test_scaled)
    else:
        y_train_scores = model.decision_function(X_train_scaled)
        y_test_scores = model.decision_function(X_test_scaled)
    classes = np.unique(Y_train)
    Y_train_bin = label_binarize(Y_train, classes=classes)
    Y_test_bin = label_binarize(Y_test, classes=classes)
    model_train_rocauc_score = roc_auc_score(Y_train_bin, y_train_scores, multi_class="ovr", average="weighted")
    model_test_rocauc_score = roc_auc_score(Y_test_bin, y_test_scores, multi_class="ovr", average="weighted")

    results[name] = model_test_accuracy

    print(f"\n{name}")

    print('Model performance for Training set')
    print("- Accuracy: {:.4f}".format(model_train_accuracy))
    print('- F1 score: {:.4f}'.format(model_train_f1))
    print('- Precision: {:.4f}'.format(model_train_precision))
    print('- Recall: {:.4f}'.format(model_train_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_train_rocauc_score))
    
    print('----------------------------------')
    
    print('Model performance for Test set')
    print('- Accuracy: {:.4f}'.format(model_test_accuracy))
    print('- F1 score: {:.4f}'.format(model_test_f1))
    print('- Precision: {:.4f}'.format(model_test_precision))
    print('- Recall: {:.4f}'.format(model_test_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_test_rocauc_score))

    print('='*35)
    print('\n')


plt.figure(figsize=(8,5))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette="viridis")
plt.xticks(rotation=30)
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.show()


tune_models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42, class_weight=class_weight_dict),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True, random_state=42, class_weight=class_weight_dict),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight=class_weight_dict),
    "Random Forest": RandomForestClassifier(random_state=42, class_weight=class_weight_dict),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}


param_grids = {
    "Logistic Regression": {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ['l1', 'l2', 'elasticnet', 'none'],
        "solver": ['saga'],   
        "l1_ratio": [0, 0.25, 0.5, 0.75, 1]  
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"]
    },
    "SVM": {
        "C": [0.01, 0.1, 1, 10, 100],
        "kernel": ["linear", "rbf", "poly", "sigmoid"],
        "gamma": ["scale", "auto"],
        "degree": [2, 3, 4]  
    },
    "Decision Tree": {
        "max_depth": [None, 5, 10, 20, 30, 50],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 10],
        "criterion": ["gini", "entropy"]
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False]
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200, 300, 500],
        "learning_rate": [0.001, 0.01, 0.1, 0.5, 1],
        "algorithm": ["SAMME", "SAMME.R"]
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 10],
        "subsample": [0.6, 0.8, 1.0],
        "max_features": ["sqrt", "log2", None]
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 10],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0, 0.1, 0.2, 0.5],
        "reg_alpha": [0, 0.01, 0.1, 1],  # L1 regularization
        "reg_lambda": [1, 1.5, 2]       # L2 regularization
    }
}

best_results = {}

for name, model in tune_models.items():
    print(f"\nTuning hyperparameters for {name}...")
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        cv=3,                 
        scoring='accuracy',   
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X_train_scaled, Y_train.values.ravel()) 

    print(f"Best parameters for {name}: {grid.best_params_}")
    print(f"Best cross-validation accuracy: {grid.best_score_:.4f}")
    
    Y_test_pred = grid.predict(X_test_scaled)
    test_accuracy = accuracy_score(Y_test, Y_test_pred)
    test_f1 = f1_score(Y_test, Y_test_pred, average='weighted')
    print(f"Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")
    
    best_results[name] = {
        "best_params": grid.best_params_,
        "cv_score": grid.best_score_,
        "test_accuracy": test_accuracy,
        "test_f1": test_f1
    }

plt.figure(figsize=(8,5))
sns.barplot(x=list(best_results.keys()), y=[best_results[m]["test_accuracy"] for m in best_results], palette="magma")
plt.ylabel("Test Accuracy")
plt.title("Hyperparameter Tuning: Model Comparison")
plt.show()

classes = np.unique(Y_train) 
n_classes = len(classes)
Y_test_bin = label_binarize(Y_test, classes=range(n_classes))

auc_models = [
    ('Logistic Regression', LogisticRegression(max_iter=500, multi_class='ovr')),
    ('KNN', KNeighborsClassifier(n_neighbors=5)),
    ('SVM', SVC(probability=True, kernel='rbf')),
    ('Decision Tree', DecisionTreeClassifier(max_depth=10)),
    ('Random Forest', RandomForestClassifier(n_estimators=200, class_weight='balanced')),
    ('AdaBoost', AdaBoostClassifier(n_estimators=200)),
    ('Gradient Boosting', GradientBoostingClassifier(n_estimators=200)),
    ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
]

plt.figure(figsize=(12, 8))
for label, model in auc_models:
    model.fit(X_train_scaled, Y_train)
    
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    else:
        y_score = model.decision_function(X_test)

    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(Y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.plot(fpr["micro"], tpr["micro"],
             label=f'{label} (AUC = {roc_auc["micro"]:.2f})')

plt.plot([0, 1], [0, 1], 'r--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Specificity (False Positive Rate)')
plt.ylabel('Sensitivity (True Positive Rate)')
plt.title('ROC Curves (One-vs-Rest, Micro-average)')
plt.legend(loc="lower right")
plt.show()