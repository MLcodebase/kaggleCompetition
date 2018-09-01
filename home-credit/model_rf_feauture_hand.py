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
parser.add_argument('--feature-rate',default=0.75 ,type=float,help='percent of feature uesed to train')

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

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation

train_x, test_x, train_y, test_y = train_test_split(train,labels,test_size=0.2,random_state=5)

param = {'random_state':[1,2,3,4,5],
         'n_estimators':[50],
#         'min_samples_leaf':[1,2,3],
#         'min_samples_split':[2,3,4],
         'max_depth':[8,16,32],
         'oob_score':[True]}
rf = RandomForestClassifier()
clf = GridSearchCV(rf, param,cv=5, n_jobs=-1, verbose=1, scoring="roc_auc")
clf.fit(train_x, train_y)
clf.best_params_

#rf = RandomForestClassifier(random_state=9,n_estimators=25,max_depth=8,oob_score=True)
#rf_scores = cross_validation.cross_val_score(rf, train, labels, cv=5)
#rf_scores
