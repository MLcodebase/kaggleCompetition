import pandas as pd
train_test = pd.read_csv('data/fill_age-cls.csv')
train_test = train_test.drop(['corseTitle','Age','Name2_new','Ticket_Num','Ticket_Letter'],axis=1)
train_test = pd.get_dummies(train_test,columns=['Cabin','Sex','fineTitle','Embarked'])
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
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=142)

from pyfm import pylibfm
fm = pylibfm.FM(k1=False,validation_size=0.005,num_factors=15, num_iter=150, verbose=True, task="classification", initial_learning_rate=0.001,init_stdev=0.002,learning_rate_schedule="optimal")
#fm.fit(X_train,y_train)
fm.fit(data,y)
import sklearn.metrics
import numpy as np
#sklearn.metrics.precision_score(y_train,np.round(fm.predict(X_train)))
#sklearn.metrics.precision_score(y_test,np.round(fm.predict(X_test)))
sklearn.metrics.precision_score(y,np.round(fm.predict(data)))
test = pd.read_csv('%s/%s' % ('data/', 'test.csv'))
test_data = scipy.sparse.csr_matrix(test_data_X_sd)
test['Survived_f'] = np.round(fm.predict(test_data))
test['Survived'] = test["Survived_f"].astype(int)
test[['PassengerId','Survived']].set_index('PassengerId').to_csv('data/sub9.csv')
