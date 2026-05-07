import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA

from sklearn.model_selection import RandomizedSearchCV



df = pd.read_csv(r"C:\\Users\\krish\\OneDrive\\Desktop\\MiniProject\\Regression\\cardekho_imputated.csv")

## UNDERSTANDING DATASET
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
    ## Storing All Different Types Of Features:(Numerical, Categorical, Discrete, Continous)
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


    ## REMOVING UNNECESSARY COLUMNS:
df.drop('Unnamed: 0', axis=1, inplace=True)
df.drop('brand', axis=1, inplace=True)
df.drop('car_name', axis=1, inplace=True)

EDA(df=df)


## MULTIVARIATE OUTLIER DETECTION THROUGH MAHALANOBIS DISTANCE
outlier_df = df.copy()
outlier_df.drop(['model', 'seller_type', 'fuel_type', 'transmission_type'], axis=1, inplace=True, errors='ignore')
print(outlier_df.head(10))

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
        # for calculating shape created this new variable
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
print(df.shape)

df.drop('Mahalanobis', axis=1, inplace=True)
print(df.head())





## SEPERATION OF INDEPENDENT AND DEPENDENT FEATURES(X,Y)) -> FOR ENCODING, SCALING, TRAINTESTSPLIT
X = df.drop('selling_price', axis=1)
Y = df['selling_price']


## TRAIN TEST SPLIT
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")



## FEATURE ENCODING AND SCALING -> (ONEHOTENCODER(Categorical Features), STANDARDSCALAR(Numerical Features), COLUMN TRANSFORM(For Handling Multiple Transformations All At Once))
# ## LABEL ENCODING WHEN THERE ARE MANY UNIQUE VALUES IN THE FEATURE
label_encoder = LabelEncoder()
X['model'] = label_encoder.fit_transform(X["model"])


## ONEHOT ENCODING FOR COLUMNS WHICH HAD LESSER UNIQUE VALUES AND NOT ORDINAL LIKE df['model']
# Create Column Transformer with 3 types of transformers
numerical_features = X_train.select_dtypes(exclude="object").columns
onehot_columns = ['seller_type','fuel_type','transmission_type']

numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", oh_transformer, onehot_columns),
        ("StandardScaler", numeric_transformer, numerical_features)
        
    ],remainder='passthrough'   
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"Processed training shape: {X_train_processed.shape}")
print(f"Processed test shape: {X_test_processed.shape}")




# MODEL TRAINING, MODEL EVALUATION, HYPERPARAMETER TUNING, RETRAINING WITH BEST PARAMETERS (MAKE FUNCTIONS FOR ALL), PLOTTING OF EVALUATION METRICS

## MODEL EVALUATION METRICS:
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square



models = {
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Ada Boost Regressor": AdaBoostRegressor(),
    "Gradient Boosting Regressor": GradientBoostingRegressor(),
    "Xgboost Regressor":XGBRegressor()
}

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train_processed, Y_train) 

    
    y_train_pred = model.predict(X_train_processed)
    y_test_pred = model.predict(X_test_processed)
    
    model_train_mae , model_train_rmse, model_train_r2 = evaluate_model(Y_train, y_train_pred)

    model_test_mae , model_test_rmse, model_test_r2 = evaluate_model(Y_test, y_test_pred)

    
    print(list(models.keys())[i])
    
    print('Model performance for Training set')
    print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
    print("- R2 Score: {:.4f}".format(model_train_r2))

    print('----------------------------------')
    
    print('Model performance for Test set')
    print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
    print("- R2 Score: {:.4f}".format(model_test_r2))
    
    print('='*35)
    print('\n')

    # pca = PCA(n_components=2)
    # reduced = pca.fit_transform(X_test)
    # plt.scatter(reduced[:,0], reduced[:,1], c=y_test_pred, cmap='viridis')
    # plt.show()



# HYPERPARAMETER TUNING:

dt_params = {
    'max_depth': [4, 6, 8, 10, 12], 
    'min_samples_split': [50, 100, 150, 200],  
    'min_samples_leaf': [20, 30, 50, 75], 
    'max_features': [0.6, 0.8, 'sqrt', 'log2'], 
    'criterion': ['squared_error', 'absolute_error'],
    'ccp_alpha': [0.0, 0.005, 0.01, 0.02, 0.03] 
}

rf_params = {
    'n_estimators': [100, 200, 300], 
    'max_depth': [6, 8, 12, 15, None], 
    'min_samples_split': [5, 10, 20, 50],  
    'min_samples_leaf': [2, 5, 10, 20], 
    'max_features': [0.6, 0.8, 'sqrt', 'log2'],
    'bootstrap': [True, False],  
    'max_samples': [0.8, 0.9, None] 
}

adaboost_params = {
    "n_estimators": [50, 100, 150, 200], 
    "learning_rate": [0.01, 0.1, 0.5, 1.0],  
    "loss": ['linear', 'square', 'exponential'],
    "random_state": [42]  
}

gradientboost_params = {
    "loss": ['squared_error', 'huber', 'absolute_error'],
    "learning_rate": [0.01, 0.05, 0.1, 0.2], 
    "n_estimators": [100, 200, 300, 500], 
    "max_depth": [3, 4, 6, 8, 10], 
    "min_samples_split": [10, 20, 50, 100],  
    "min_samples_leaf": [5, 10, 20],
    "subsample": [0.8, 0.9, 1.0],  
    "max_features": [0.8, 'sqrt', 'log2'],  
    "validation_fraction": [0.1],  
    "n_iter_no_change": [10] 
}

xgboost_params = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2],  
    "max_depth": [3, 4, 6, 8, 10],  
    "n_estimators": [100, 200, 300, 500], 
    "colsample_bytree": [0.6, 0.8, 0.9, 1.0],  
    "subsample": [0.8, 0.9, 1.0], 
    "reg_alpha": [0, 0.1, 0.5, 1.0],  
    "reg_lambda": [1, 1.5, 2.0],
    "min_child_weight": [1, 3, 5],  
    "gamma": [0, 0.1, 0.2],  
    "random_state": [42]  
}


randomcv_models = [("DT", DecisionTreeRegressor(), dt_params),
                   ("RF", RandomForestRegressor(), rf_params),
                   ("AB", AdaBoostRegressor(), adaboost_params),
                   ("GB", GradientBoostingRegressor(), gradientboost_params),
                   ("XGB", XGBRegressor(), xgboost_params)
                   ]

model_param = {}
for name, model, params in randomcv_models:
    random = RandomizedSearchCV(estimator=model,
                                   param_distributions=params,
                                   n_iter=100,
                                   cv=3,
                                   verbose=2,
                                   n_jobs=-1)
    random.fit(X_train_processed, Y_train)
    model_param[name] = random.best_params_

for model_name in model_param:
    print(f"---------------- Best Params for {model_name} -------------------")
    print(model_param[model_name])
    

dt_max_depth = model_param['DT']['max_depth']
dt_min_samples_split = model_param['DT']['min_samples_split']  
dt_min_samples_leaf = model_param['DT']['min_samples_leaf']
dt_max_features = model_param['DT']['max_features']
dt_criterion = model_param['DT']['criterion']
dt_ccp_alpha = model_param['DT']['ccp_alpha']

rf_n_estimators = model_param['RF']['n_estimators']
rf_max_depth = model_param['RF']['max_depth']
rf_min_samples_split = model_param['RF']['min_samples_split']
rf_min_samples_leaf = model_param['RF']['min_samples_leaf']
rf_max_features = model_param['RF']['max_features']
rf_bootstrap = model_param['RF']['bootstrap']
rf_max_samples = model_param['RF']['max_samples']

ab_n_estimators = model_param['AB']['n_estimators']
ab_learning_rate = model_param['AB']['learning_rate']
ab_loss = model_param['AB']['loss']
ab_random_state = model_param['AB']['random_state']

gb_loss = model_param['GB']['loss']
gb_learning_rate = model_param['GB']['learning_rate']
gb_n_estimators = model_param['GB']['n_estimators']
gb_max_depth = model_param['GB']['max_depth']
gb_min_samples_split = model_param['GB']['min_samples_split']
gb_min_samples_leaf = model_param['GB']['min_samples_leaf']
gb_subsample = model_param['GB']['subsample']
gb_max_features = model_param['GB']['max_features']
gb_validation_fraction = model_param['GB']['validation_fraction']
gb_n_iter_no_change = model_param['GB']['n_iter_no_change']

xgb_learning_rate = model_param['XGB']['learning_rate']
xgb_max_depth = model_param['XGB']['max_depth']
xgb_n_estimators = model_param['XGB']['n_estimators']
xgb_colsample_bytree = model_param['XGB']['colsample_bytree']
xgb_subsample = model_param['XGB']['subsample']
xgb_reg_alpha = model_param['XGB']['reg_alpha']
xgb_reg_lambda = model_param['XGB']['reg_lambda']
xgb_min_child_weight = model_param['XGB']['min_child_weight']
xgb_gamma = model_param['XGB']['gamma']
xgb_random_state = model_param['XGB']['random_state']


## RETRAINING THE MODELS WITH BEST PARAMETERS

models = {
    "Decision Tree Regressor": DecisionTreeRegressor(
        criterion=dt_criterion,
        max_depth=dt_max_depth,
        min_samples_split=dt_min_samples_split,
        min_samples_leaf=dt_min_samples_leaf,
        max_features=dt_max_features,
        ccp_alpha=dt_ccp_alpha
    ),
    
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        min_samples_split=rf_min_samples_split,
        min_samples_leaf=rf_min_samples_leaf,
        max_features=rf_max_features,
        bootstrap=rf_bootstrap,
        max_samples=rf_max_samples,
        n_jobs=-1
    ),
    
    "Adaboost": AdaBoostRegressor(
        n_estimators=ab_n_estimators,
        learning_rate=ab_learning_rate,
        loss=ab_loss,
        random_state=ab_random_state
    ),
    
    "GradientBoost Regressor": GradientBoostingRegressor(
        loss=gb_loss,
        learning_rate=gb_learning_rate,
        n_estimators=gb_n_estimators,
        max_depth=gb_max_depth,
        min_samples_split=gb_min_samples_split,
        min_samples_leaf=gb_min_samples_leaf,
        subsample=gb_subsample,
        max_features=gb_max_features,
        validation_fraction=gb_validation_fraction,
        n_iter_no_change=gb_n_iter_no_change
    ),
    
    "Xgboost Regressor": XGBRegressor(
        learning_rate=xgb_learning_rate,
        max_depth=xgb_max_depth,
        n_estimators=xgb_n_estimators,
        colsample_bytree=xgb_colsample_bytree,
        subsample=xgb_subsample,
        reg_alpha=xgb_reg_alpha,
        reg_lambda=xgb_reg_lambda,
        min_child_weight=xgb_min_child_weight,
        gamma=xgb_gamma,
        random_state=xgb_random_state
    )
}

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train_processed, Y_train)

    y_train_pred = model.predict(X_train_processed)
    y_test_pred = model.predict(X_test_processed)

    model_train_mae , model_train_rmse, model_train_r2 = evaluate_model(Y_train, y_train_pred)

    model_test_mae , model_test_rmse, model_test_r2 = evaluate_model(Y_test, y_test_pred)
    
    print(list(models.keys())[i])
    
    print('Model performance for Training set')
    print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
    print("- R2 Score: {:.4f}".format(model_train_r2))

    print('----------------------------------')
    
    print('Model performance for Test set')
    print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
    print("- R2 Score: {:.4f}".format(model_test_r2))
    
    print('='*35)
    print('\n')

    # pca = PCA(n_components=2)
    # reduced = pca.fit_transform(X_test)
    # plt.scatter(reduced[:,0], reduced[:,1], c=y_test_pred, cmap='viridis')
    # plt.show()
    



## FEATURE IMPORTANCES

def get_feature_names_from_preprocessor(preprocessor):
    feature_names = []
    
    # OneHot encoded features
    onehot_features = preprocessor.named_transformers_['OneHotEncoder'].get_feature_names_out(['seller_type', 'fuel_type', 'transmission_type'])
    feature_names.extend(onehot_features)
    
    # Numerical features (same names)
    numerical_features = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats', 'model']
    feature_names.extend(numerical_features)
    
    return feature_names

correct_feature_names = get_feature_names_from_preprocessor(preprocessor)

for name, model in models.items():
    feature_importance = pd.DataFrame({
        'feature': correct_feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 Most Important Features of {name}")
    print(feature_importance.head(15))
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
    plt.title(f'Top 10 Feature Importances - {name}')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()
    

    