# Basic imports
import pandas as pd
import numpy as np

# Plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Local imports
from dataset import *

rcParams['patch.force_edgecolor'] = True
rcParams['patch.facecolor'] = 'b'

# --- Read data --- #
cols = ['Age', 'Workclass', 'Fnlwgt', 'Education', 'Education num', 
        'Marital status', 'Occupation', 'Relationship', 'Race', 'Sex', 
        'Capital gain', 'Capital loss', 'Hours per week', 'Native country', 
        'Target']

df = pd.read_csv('adult.data', header = None)
df.columns = cols
df = Dataset(df, 'Target')

# --- Find and treat missing values --- #
null_vals = df.data.isnull().sum()

# Check for wrong data type in columns
types_ok = df.check_types()

# Replace ' ?' values
df.data['Workclass'].replace(to_replace = ' ?', value = 'Unknown', 
       inplace = True)
df.data['Occupation'].replace(to_replace = ' ?', value = 'Other', 
       inplace = True)

# Fill 'Workclass' == 'Unknown' with mode value
df.data['Workclass'].replace(to_replace = 'Unknown', 
       value = df.data['Workclass'].mode().values[0], inplace = True)

# --- Explore data --- #

# One dimension plots
sns.distplot(df.data['Age'], bins = 15, kde = False)
sns.countplot(df.data['Workclass'])
sns.kdeplot(df.data['Fnlwgt'])
sns.countplot(df.data['Education'])
sns.distplot(df.data['Education num'])
sns.countplot(df.data['Marital status'])
sns.countplot(df.data['Occupation']) 
sns.countplot(df.data['Relationship'])
sns.countplot(df.data['Race'])
sns.countplot(df.data['Sex'])
sns.kdeplot(df.data['Capital gain'])
sns.kdeplot(df.data['Capital loss'])
sns.distplot(df.data['Hours per week'], bins = 10, kde = False)
sns.countplot(df.data['Native country'])
sns.countplot(df.data['Target'])

# Two dimension plots
fig = sns.FacetGrid(data = df.data, col = 'Target')
fig.map(sns.countplot, 'Race')

fig = sns.FacetGrid(data = df.data, col = 'Education', col_wrap = 3)
fig.map(sns.boxplot, 'Education num')

fig = sns.FacetGrid(data = df.data, col = 'Target')
fig.map(sns.kdeplot, 'Hours per week')

fig = sns.FacetGrid(data = df.data, col = 'Target')
fig.map(sns.kdeplot, 'Hours per week')

# Correlations
df.encoder()
df.plot_corr()

""" These attributes show the stronger correlations to the target:
    Age
    Education num
    Marital status
    Relationship
    Sex
    Capital gain
    Capital loss
    Hours per week
"""

# --- Attribute selection --- #
rfe_info = df.rfe(df.encoded_data, 5, 10, 'accuracy', 3) 
#rfe_info = df.best_rfe([2, 4, 6], 5, 'accuracy', 3)

# --- Write to file --- #
cols = np.where(rfe_info.support_ == True)[0]
train_data = df.data.drop(columns = df.target_col).copy()
train_data = train_data.iloc[:,cols]
train_data[df.target_col] = df.data[df.target_col]

train_data.to_csv('train.csv')










