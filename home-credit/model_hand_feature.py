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
parser.add_argument('--model-type',  default='lgbm',type=str,help='model type')
parser.add_argument('--feature-rate',default=0.9 ,type=float,help='percent of feature uesed to train')

args = parser.parse_args()

fi = pd.read_csv('data/feature_hand_importance.csv')
total_feature = fi.shape[0]
columns_drop = list(fi.sort_values('importance')['feature'])
columns_drop = columns_drop[0:int(total_feature*(1-args.feature_rate))]
train = pd.read_csv('data/train-hand-feature.csv')
test = pd.read_csv('data/test-hand-feature.csv')

submission, fi, metrics = model(train, test, drop_columns=columns_drop, model_type = args.model_type)   

'''
submission.to_csv('data/feature_hand_lgbm2.csv',index=False)
fi.to_csv('data/feature_hand_importance2.csv')
metrics.to_csv('data/feature_hand_metric2.csv')
'''
print 'feature importance:'
print fi
print 'final metrics:'
print metrics