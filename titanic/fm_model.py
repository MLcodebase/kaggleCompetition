import pandas as pd
import sklearn.metrics
import numpy as np
import scipy.sparse
from sklearn.preprocessing import StandardScaler
from pyfm import pylibfm
from sklearn import model_selection
from sklearn.base import BaseEstimator

train_test = pd.read_csv('data/fill_age-cls.csv')
train_test = train_test.drop(['corseTitle','Age','Name2_new','Ticket_Num','Ticket_Letter'],axis=1)
train_test = pd.get_dummies(train_test,columns=['Cabin','Sex','fineTitle','Embarked','corseAge'])
train_data = train_test[:891]
test_data = train_test[891:]
train_data_X = train_data.drop(['Survived'],axis=1)
train_data_Y = train_data['Survived']
test_data_X = test_data.drop(['Survived'],axis=1)

ss = StandardScaler()
ss.fit(train_data_X)
train_data_X_sd = ss.transform(train_data_X)
test_data_X_sd  = ss.transform(test_data_X)

data = scipy.sparse.csr_matrix(train_data_X_sd)
y = train_data_Y.values
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=142)

class fm_sk(BaseEstimator):
  def __init__(self,
                 num_factors=10,
                 num_iter=1,
                 k0=True,
                 k1=True,
                 init_stdev=0.1,
                 validation_size=0.01,
                 learning_rate_schedule="optimal",
                 initial_learning_rate=0.01,
                 power_t=0.5,
                 t0=0.001,
                 task='classification',
                 verbose=True,
                 shuffle_training=True,
                 seed = 28):
      super(BaseEstimator,self).__init__()
      self.num_factors=num_factors
      self.num_iter=num_iter
      self.k0=k0
      self.k1=k1
      self.init_stdev=init_stdev
      self.validation_size=validation_size
      self.learning_rate_schedule=learning_rate_schedule
      self.initial_learning_rate=initial_learning_rate
      self.power_t=power_t
      self.t0=t0
      self.task=task
      self.verbose=verbose
      self.shuffle_training=shuffle_training
      self.seed=seed
      self.fm = pylibfm.FM(num_factors=self.num_factors,num_iter=self.num_iter,k0=self.k0,k1=self.k1,init_stdev=self.init_stdev,validation_size=self.validation_size,learning_rate_schedule=self.learning_rate_schedule,initial_learning_rate=self.initial_learning_rate,power_t=self.power_t,t0=self.t0,task=self.task,verbose=self.verbose,shuffle_training=self.shuffle_training,seed = self.seed)
  def fit(self,x,y):
    x = scipy.sparse.csr_matrix(x)
    y = y.values
    self.fm.fit(x,y)
  def score(self,x,y):
    x = scipy.sparse.csr_matrix(x)
    y = y.values
    return sklearn.metrics.precision_score(y,np.round(self.fm.predict(x)))

class fm_sk(BaseEstimator):
  def __init__(self):
      self.fm = pylibfm.FM(k1=False,validation_size=0.005,num_factors=10, num_iter=8, verbose=True, task="classification", initial_learning_rate=0.001,init_stdev=0.002,learning_rate_schedule="optimal")
  def fit(self,x,y):
    x = scipy.sparse.csr_matrix(x)
    y = y.values
    self.fm.fit(x,y)
  def score(self,x,y):
    x = scipy.sparse.csr_matrix(x)
    y = y.values
    return sklearn.metrics.precision_score(y,np.round(self.fm.predict(x)))

fm_wrapper = fm_sk()
cv = model_selection.KFold(n_splits=5, random_state=3)
model_selection.cross_val_score(fm_wrapper,train_data_X_sd,train_data_Y,cv=cv)

#fm.fit(X_train,y_train)
fm = pylibfm.FM(k1=False,validation_size=0.005,num_factors=10, num_iter=80, verbose=True, task="classification", initial_learning_rate=0.001,init_stdev=0.002,learning_rate_schedule="optimal")
fm.fit(data,y)

#sklearn.metrics.precision_score(y_train,np.round(fm.predict(X_train)))
#sklearn.metrics.precision_score(y_test,np.round(fm.predict(X_test)))
sklearn.metrics.precision_score(y,np.round(fm.predict(data)))
test = pd.read_csv('%s/%s' % ('data/', 'test.csv'))
test_data = scipy.sparse.csr_matrix(test_data_X_sd)
test['Survived_f'] = np.round(fm.predict(test_data))
test['Survived'] = test["Survived_f"].astype(int)
test[['PassengerId','Survived']].set_index('PassengerId').to_csv('data/sub9.csv')
