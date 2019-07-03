# Basic imports
import numpy as np
import pandas as pd
import ast

# General sklearn imports
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

# Models imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier

# Local imports
from dataset import *

# --- Read data --- #
df = Dataset(pd.read_csv('train.csv'), 'Target')
df.encoder()
atr = df.scale_atr(df.encoded_data)

target = atr[df.target_col].values
atr = atr.drop(columns = df.target_col).values

# --- Create models --- #
grid = pd.read_csv('forestGrid.csv').sort_values(by = 'mean_test_score', 
                  ascending = False)
param = ast.literal_eval(grid.loc[0,'params'])
forest = RandomForestClassifier(criterion = param['criterion'],
                                max_depth = param['max_depth'],
                                max_features = param['max_features'],
                                min_weight_fraction_leaf = param['min_weight_fraction_leaf'],
                                n_estimators = param['n_estimators'])

grid = pd.read_csv('KnnGrid.csv').sort_values(by = 'mean_test_score', 
                  ascending = False)
param = ast.literal_eval(grid.loc[0,'params'])
knn = KNN(algorithm = param['algorithm'],
          leaf_size = param['leaf_size'],
          n_neighbors = param['n_neighbors'],
          p = param['p'],
          weights = param['weights'],)

grid = pd.read_csv('XGBgrid.csv').sort_values(by = 'mean_test_score', 
                  ascending = False)
param = ast.literal_eval(grid.loc[0,'params'])
xgb_model = XGBClassifier(booster = param['booster'],
                          gamma = param['gamma'],
                          learning_rate = param['learning_rate'],
                          max_delta_step = param['max_delta_step'],
                          max_depth = param['max_depth'],
                          n_estimators = param['n_estimators'])

grid = pd.read_csv('LogGrid.csv').sort_values(by = 'mean_test_score', 
                  ascending = False)
param = ast.literal_eval(grid.loc[0,'params'])
log_model = LogisticRegression(C = param['C'],
                               fit_intercept = param['fit_intercept'],
                               max_iter = param['max_iter'],
                               solver = param['solver'],
                               tol = param['tol'],)

grid = pd.read_csv('RidGrid.csv').sort_values(by = 'mean_test_score', 
                  ascending = False)
param = ast.literal_eval(grid.loc[0,'params'])
rid_model = RidgeClassifier(alpha = param['alpha'],
                            fit_intercept = param['fit_intercept'],
                            solver = param['solver'],
                            tol = param['tol'])

# --- Cross-validate models --- #
models = [forest, knn, xgb_model, log_model, rid_model]
model_names = ['RandFor', 'KNN', 'Xgb', 'LogReg', 'Ridg']
accuracy = []
for m in models:
    accuracy.append(np.mean(cross_val_score(m, atr, target, cv = 10, 
                                    scoring = 'accuracy')))
