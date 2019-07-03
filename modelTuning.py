# Basic imports
import pandas as pd
import numpy as np

# General SkLearn imports
from sklearn.model_selection import GridSearchCV

# Estimator imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier

# Local imports
from dataset import *

def tuning(model, xdata, ydata, param, nv, jobs, file_name):
    tuner = GridSearchCV(estimator = model, param_grid = param, 
                         scoring = "accuracy", cv = nv, verbose = 3, 
                         n_jobs = jobs)
    tuner.fit(xdata, ydata)
    all_runs = pd.DataFrame(data = tuner.cv_results_)
    all_runs.to_csv(file_name)
    
    return all_runs, tuner.best_params_

# --- Read data --- #
df = Dataset(pd.read_csv('train.csv', index_col = 0), 'Target')
df.encoder()
atr = df.encoded_data.drop(columns = df.target_col).values
target = df.encoded_data[df.target_col].values

# --- Set up models hyperparameters --- #
rf_param = {"n_estimators": [100, 500, 1000],
            "criterion": ["gini", "entropy"],
            "max_depth": [8, 18, None],
            "min_weight_fraction_leaf": [0, 0.5],
            "max_features": ["auto", "log2", None],
            "class_weight": [None, "balanced"]}
knn_param = {'n_neighbors': [2, 5, 10, 20],
             'weights': ['uniform', 'distance'],
             'algorithm': ['auto', 'ball_tree', 'kd_tree'],
             'leaf_size': [5, 15, 30],
             'p': [1, 2, 3]}
xgb_param = {'booster': ['gbtree', 'gblinear'],
             'gamma': [0, 0.3],
             'learning_rate': [0.05, 0.1, 0.3],
             'max_delta_step': [0, 2],
             'max_depth': [3, 5, 10],
             'n_estimators': [100, 500]}
log_param = {'tol': [1e-4, 1e-3],
             'C': [1.0, 2.0, 4.0],
             'fit_intercept': [True, False],
             'class_weight': [None, 'balanced'],
             'solver': ['newton-cg', 'sag', 'lbfgs', 'liblinear'],
             'max_iter': [100, 300]} 
rid_param = {'alpha': [1.0, 2.0, 3.0],
             'fit_intercept': [True, False],
             'tol': [1e-3, 1e-4],
             'solver': ['svd', 'cholesky', 'sparse_cg', 'sag']}

# --- Tune models --- #
rf_grid, rf_best_params = tuning(RandomForestClassifier(), atr, target, 
                                 rf_param, 5, 3, 'forestGrid.csv')

knn_grid, knn_best_params = tuning(KNN(), atr, target, 
                                   knn_param, 5, 3, 'KnnGrid.csv')

xgb_grid, xgb_best_params = tuning(XGBClassifier(), atr, target, 
                                 xgb_param, 5, 3, 'XgbGrid.csv')

log_grid, log_best_params = tuning(LogisticRegression(), atr, target, 
                                 log_param, 5, 3, 'LogGrid.csv')

rid_grid, rid_best_params = tuning(RidgeClassifier(), atr, target, 
                                 rid_param, 5, 3, 'RidGrid.csv')


