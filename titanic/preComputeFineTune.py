import pandas as pd
import numpy as np
train_test = pd.read_csv('data/pre_proc.csv')
train_test['Cabin_Letter'] = train_test['Cabin'].apply(lambda x: x[0] if not isinstance(x,float) else x)
train_test['Name2_new'] = train_test['Name2_new'].apply(lambda x: np.nan if x=='one' else x)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for x in train_test.columns:
    if train_test[x].dtypes=='object':
       train_test[x]=le.fit_transform(train_test[x])
