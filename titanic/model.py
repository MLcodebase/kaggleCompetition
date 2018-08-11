import pandas as pd
train_test = pd.read_csv('data/fill_age-cls.csv')
train_test = train_test.drop(['corseTitle'],axis=1)
train_test = pd.get_dummies(train_test,columns=['Cabin','Sex','fineTitle','Ticket_Letter','Embarked','Name2_new','Age'])
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

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=150,min_samples_leaf=3,max_depth=6,oob_score=True)
from sklearn import cross_validation
rf_scores = cross_validation.cross_val_score(rf, train_data_X, train_data_Y, cv=5)
rf.fit(train_data_X,train_data_Y)

from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
lr = LogisticRegression()
param = {'C':[0.001,0.01,0.1,1,10], "max_iter":[100,250]}
clf = GridSearchCV(lr, param,cv=5, n_jobs=-1, verbose=1, scoring="roc_auc")
clf.fit(train_data_X_sd, train_data_Y)
clf.grid_scores_
clf.best_params_
lr = LogisticRegression(C=0.01,max_iter=100)
lr.fit(train_data_X_sd, train_data_Y)

from sklearn import svm
svc = svm.SVC()
clf = GridSearchCV(svc,param,cv=5,n_jobs=-1,verbose=1,scoring="roc_auc")
clf.fit(train_data_X_sd,train_data_Y)
clf.best_params_
svc = svm.SVC(C=1,max_iter=250)
svc.fit(train_data_X_sd,train_data_Y)
#svc.predict(test_data_X_sd)

from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier(learning_rate=0.7,max_depth=6,n_estimators=100,min_samples_leaf=2)
gbdt.fit(train_data_X,train_data_Y)

import xgboost as xgb
xgb_model = xgb.XGBClassifier(n_estimators=150,min_samples_leaf=3,max_depth=6)
xgb_model.fit(train_data_X,train_data_Y)

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=0.1,max_iter=100)

import xgboost as xgb
xgb_model = xgb.XGBClassifier(max_depth=6,min_samples_leaf=2,n_estimators=100,num_round = 5)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200,min_samples_leaf=2,max_depth=6,oob_score=True)
from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier(learning_rate=0.1,min_samples_leaf=2,max_depth=6,n_estimators=100)
vot = VotingClassifier(estimators=[('lr', lr), ('rf', rf),('gbdt',gbdt),('xgb',xgb_model)], voting='hard')
vot_scores = cross_validation.cross_val_score(vot, train_data_X, train_data_Y, cv=5)
vot.fit(train_data_X_sd,train_data_Y)
test = pd.read_csv('%s/%s' % ('data/', 'test.csv'))
test["Survived_f"] = vot.predict(test_data_X_sd)
test.to_csv('data/vote.csv',index=False)
test[['PassengerId','Survived']].set_index('PassengerId').to_csv('data/sub1.csv')

