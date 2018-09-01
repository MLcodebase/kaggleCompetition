import pandas as pd
import numpy as np
# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns
# Suppress warnings from pandas
import warnings
warnings.filterwarnings('ignore')
import gc
from helpers import *
import argparse

parser = argparse.ArgumentParser(description='Train face network')
parser.add_argument('--model-type',  default='rf',type=str,help='model type')
parser.add_argument('--feature-rate',default=0.9 ,type=float,help='percent of feature uesed to train')

args = parser.parse_args()

fi = pd.read_csv('data/feature_hand_importance.csv')
total_feature = fi.shape[0]
columns_drop = list(fi.sort_values('importance')['feature'])
columns_drop = columns_drop[0:int(total_feature*(1-args.feature_rate))]
train = pd.read_csv('data/train-hand-feature.csv')
test = pd.read_csv('data/test-hand-feature.csv')

train_ids = train['SK_ID_CURR']
test_ids = test['SK_ID_CURR']
# Extract the labels for training
labels = train['TARGET']
# Remove the ids and target
train = train.drop(columns = ['SK_ID_CURR', 'TARGET'])
test = test.drop(columns = ['SK_ID_CURR'])
# One Hot Encoding
train = pd.get_dummies(train)
test  = pd.get_dummies(test)
# Align the dataframes by the columns
train, test = train.align(test, join = 'inner', axis = 1)
if columns_drop != None:
    train = train.drop(columns = columns_drop,axis = 1)
    test = test.drop(columns = columns_drop,axis = 1)

train, test = remove_missing_columns(train,test,threshold=90)
fill_nan_columns(train)
fill_nan_columns(test)
train, test = train.align(test, join = 'inner', axis = 1)
# Extract feature names
feature_names = list(train.columns)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
#train_x, test_x, train_y, test_y = train_test_split(train,labels,test_size=0.2,random_state=5)
param = {'random_state':[2],
         'n_estimators':[50],
         'min_samples_leaf':[10,20],
         'min_samples_split':[10,20],
         'max_depth':[15,20],
         'oob_score':[True]}
rf = RandomForestClassifier()
clf = GridSearchCV(rf, param,cv=4, n_jobs=5, verbose=1, scoring="roc_auc",return_train_score=True)
clf.fit(train, labels)
clf.best_params_
clf.best_score_
clf.cv_results_['mean_train_score']
clf.cv_results_['mean_test_score']
clf.cv_results_['params']

rf = RandomForestClassifier(random_state=2,n_estimators=50,max_depth=12,min_samples_leaf=20,min_samples_split=10,oob_score=True)
rf_scores = cross_val_score(rf, train, labels, scoring="roc_auc", cv=5,n_jobs=-1)
#rf_scores
