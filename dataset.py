import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

class Dataset:
    
    def __init__(self, data, target_col):
        self.data = data
        self.target_col = target_col
    
    # Preprocessing functions
    def atr_and_target(self, target_col):
        self.atr = self.data.drop(columns = target_col).copy()
        self.target = self.data[target_col].copy()

    def encoder(self):
        code_list = []
        col_names = []
        self.encoded_data = self.data.copy()
        for col in self.encoded_data.columns.values:
            if self.encoded_data[col].dtypes == object:
                coder = LabelEncoder()
                self.encoded_data[col] = coder.fit_transform(self.encoded_data[col].values)
                code_list.append(coder)
                col_names.append(col)
        self.encoder_list = pd.DataFrame(data = code_list)
        self.encoder_list.index = col_names
        
    def scale_atr(self, data):
        self.scaler = StandardScaler()
        dummy = data.drop(columns = self.target_col).copy()
        dummy = self.scaler.fit_transform(dummy)
        dummy_target = data[self.target_col].values
        self.data = pd.DataFrame(data = dummy, columns = 
                    data.drop(columns = self.target_col).columns.values)
        data[self.target_col] = dummy_target
        return data

    # Handling missing values functions
    def check_types(self):
        is_ok = []
        for col in self.data.columns.values:
            if self.data[col].dtypes == np.dtype('int64'):
                dummy = [type(x) == np.int64 for x in self.data[col].values]
            elif self.data[col].dtypes == np.dtype('float64'):
                dummy = [type(x) == np.float64 for x in self.data[col].values]
            elif self.data[col].dtypes == np.dtype('object'):
                dummy = [type(x) == str for x in self.data[col].values]
            
            if False in dummy:
                is_ok.append(False)
            else:
                is_ok.append(True)
        return is_ok

    # Data exploration functions        
    def plot_corr(self, data):
        corr = data.corr()
        sns.heatmap(corr, annot = True)    
    
    # Feature selection functions
    def rfe(self, data, n_atr, ncv, score_type, jobs):
        selector = RFECV(estimator = RandomForestClassifier(),
                         step = 1,
                         min_features_to_select = n_atr,
                         cv = ncv,
                         scoring = score_type,
                         verbose = 3,
                         n_jobs = jobs)
        atr = data.drop(columns = self.target_col).values
        target = data[self.target_col].values
        selector = selector.fit(atr, target)
        return selector
    
    def best_rfe(self, data, natr, ncv, score_type, jobs):
        scores = []
        support = []
        for num in natr:
            selection = self.rfe(data, num, ncv, score_type, jobs)
            
            pos = np.asarray(np.where(selection.support_ == True))[0]
            atr = data.drop(columns = self.target_col).values[:,pos]
            target = data[self.target_col].values
            
            model = RandomForestClassifier()
            new_score = cross_val_score(estimator = model, X = atr, y = target,
                                        scoring = score_type, cv = ncv, 
                                        n_jobs = jobs)
            scores.append(np.mean(new_score))
            support.append(selection.support_)
            
        all_select = pd.DataFrame()
        all_select['Attributes'] = support
        all_select['MeanScore'] = scores
        return all_select.sort_values(by = 'MeanScore', ascending = False)
        