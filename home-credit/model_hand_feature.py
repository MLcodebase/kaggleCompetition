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

fi = pd.read_csv('data/feature_hand_importance.csv')

columns_drop = list(fi.sort_values('importance')[:200]['feature'])

columns_drop.append('DAYS_BIRTH')

train = pd.read_csv('data/train-hand-feature.csv')
test = pd.read_csv('data/test-hand-feature.csv')

train = train.drop(columns = columns_drop,axis=1)
test  = test.drop(columns = columns_drop,axis=1)

submission, fi, metrics = model(train, test)

submission.to_csv('data/feature_hand_lgbm2.csv',index=False)
fi.to_csv('data/feature_hand_importance2.csv')
metrics.to_csv('data/feature_hand_metric2.csv')

print 'feature importance:'
print fi
print 'final metrics:'
print metrics