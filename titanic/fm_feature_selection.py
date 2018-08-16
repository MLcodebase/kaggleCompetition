import pandas as pd
train_test = pd.read_csv('data/fill_age-cls.csv')
train_test['Cabin_Letter'] = train_test['Cabin'].apply(lambda x: x[0] if not isinstance(x,float) else x)
train_test['Name2_new'] = train_test['Name2_new'].apply(lambda x: np.nan if x=='one' else x)
train_test = train_test.drop(['Age','age_nan','SibSp','Ticket_Num','fineTitle','corseTitle','Ticket_Letter'],axis=1)
train_test = pd.get_dummies(train_test,columns=['Cabin','Name2_new','Embarked','Sex'])
train_data = train_test[:891]
test_data = train_test[891:]
train_data_X = train_data.drop(['Survived'],axis=1)
train_data_Y = train_data['Survived']
test_data_X = test_data.drop(['Survived'],axis=1)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_data_X)
train_data_X_sd = ss.transform(train_data_X)
test_data_X_sd  = ss.transform(test_data_X)
import scipy.sparse
data = scipy.sparse.csr_matrix(train_data_X_sd)
y = train_data_Y.values
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=3)
from pyfm import pylibfm
fm = pylibfm.FM(seed=28,k1=False,init_stdev=0.002,validation_size=0.005,num_factors=28, num_iter=80, verbose=True, task="classification", initial_learning_rate=0.001, learning_rate_schedule="optimal")
fm.fit(X_train,y_train)
import sklearn.metrics
import numpy as np
sklearn.metrics.precision_score(y_train,np.round(fm.predict(X_train)))
sklearn.metrics.precision_score(y_test,np.round(fm.predict(X_test)))