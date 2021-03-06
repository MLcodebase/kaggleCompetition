import pandas as pd
train_test = pd.read_csv('data/fill_age-cls.csv')
train_test = train_test.drop(['corseTitle','Age'],axis=1)
train_test['Cabin_Letter'] = train_test['Cabin'].apply(lambda x: x[0] if not isinstance(x,float) else x)
#train_test = pd.get_dummies(train_test,columns=['Cabin','Sex','fineTitle','Ticket_Letter','Embarked','Name2_new','corseAge','Ticket_Num'])
train_test = pd.get_dummies(train_test,columns=['Cabin_Letter','Cabin','Sex','fineTitle','Ticket_Letter','Embarked','Name2_new','corseAge','Ticket_Num'])
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

from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
lr = LogisticRegression()
param = {'C':[0.00125,0.015,0.02], "max_iter":[20,25,30,35,40,50]}
clf = GridSearchCV(lr, param,cv=5, n_jobs=-1, verbose=1, scoring="roc_auc")
clf.fit(train_data_X_sd, train_data_Y)
clf.best_params_
from sklearn import cross_validation
lr = LogisticRegression(C=0.015,max_iter=20)
lr_scores = cross_validation.cross_val_score(lr, train_data_X, train_data_Y, cv=5)
lr_scores

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5,p=1)
from sklearn import cross_validation
knn_scores = cross_validation.cross_val_score(knn, train_data_X_sd, train_data_Y, cv=5)
knn_scores
knn.fit(train_data_X_sd, train_data_Y)
knn.score(train_data_X_sd, train_data_Y)


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
param = {'random_state':[1,2,3,4,5,6,7,8,9,10,11,12],
         'n_estimators':[25,28,30,32],
         'min_samples_leaf':[1,2,3],
         'min_samples_split':[2,3,4],
         'max_depth':[6,7,8],
         'oob_score':[True]}
clf = GridSearchCV(rf, param,cv=5, n_jobs=-1, verbose=1, scoring="roc_auc")
clf.fit(train_data_X_sd, train_data_Y)
clf.best_params_
from sklearn import cross_validation
rf = RandomForestClassifier(random_state=9,n_estimators=25,min_samples_leaf=1,min_samples_split=3,max_depth=8,oob_score=True)
rf_scores = cross_validation.cross_val_score(rf, train_data_X, train_data_Y, cv=5)
rf_scores

from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier()
param = {"criterion":["mae","friedman_mse"],
         "learning_rate":[0.09,0.1,1.1],
         "n_estimators":[95,100,105],
         "min_samples_leaf":[2],
         "max_depth":[4,5,6],
         "min_samples_split": [3],
         "subsample": [0.5, 0.6, 0.7],
         "max_features": ["auto"],
         }
from sklearn.grid_search import GridSearchCV
clf = GridSearchCV(gbdt, param,cv=5, n_jobs=-1, verbose=1)
clf.fit(train_data_X,train_data_Y)
clf.best_params_

gbdt = GradientBoostingClassifier(n_estimators=120,learning_rate=0.1,max_depth=4,min_samples_leaf=2,subsample=0.7,min_samples_split=3)
gbdt_scores = cross_validation.cross_val_score(gbdt, train_data_X, train_data_Y, cv=5)
gbdt_scores

import xgboost as xgb
xgb_model = xgb.XGBClassifier()
param = {
    "n_estimators":[120,130,140,150],
    "learning_rate":[0.045,0.05,0.055],
    "max_depth":[6,7,8],
    "reg_alpha": [0.01],
    "reg_lambda": [1],
    "min_child_weight": [1, 2],
    "subsample": [0.75],
    "colsample_bytree": [0.5],
         }
from sklearn.grid_search import GridSearchCV
clf = GridSearchCV(xgb_model, param,cv=5, n_jobs=-1, verbose=1)
clf.fit(train_data_X,train_data_Y)
clf.best_params_
xgb_model = xgb.XGBClassifier(n_estimators=160,learning_rate=0.004,max_depth=6,colsample_bytree=0.5,subsample=0.75,reg_alpha=0.01)
from sklearn import cross_validation
xg_scores = cross_validation.cross_val_score(xgb_model, train_data_X, train_data_Y, cv=5)
xg_scores

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=0.015,max_iter=20)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=9,n_estimators=25,min_samples_leaf=1,min_samples_split=3,max_depth=8,oob_score=True)
from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier(n_estimators=120,learning_rate=0.1,max_depth=4,min_samples_leaf=2,subsample=0.7,min_samples_split=3)
import xgboost as xgb
xgb_model = xgb.XGBClassifier(n_estimators=160,learning_rate=0.004,max_depth=6,colsample_bytree=0.5,subsample=0.75,reg_alpha=0.01)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5,p=1)

#vot = VotingClassifier(estimators=[('lr', lr), ('rf', rf),('gbdt',gbdt),('xgb',xgb_model),('knn',knn)], voting='hard')
vot = VotingClassifier(estimators=[('rf', rf),('gbdt',gbdt),('xgb',xgb_model)], voting='hard')
from sklearn import cross_validation
vot_scores = cross_validation.cross_val_score(vot, train_data_X, train_data_Y, cv=5)
vot_scores
vot.fit(train_data_X_sd,train_data_Y)

test = pd.read_csv('%s/%s' % ('data/', 'test.csv'))
test["Survived_f"] = vot.predict(test_data_X_sd)
test['Survived'] = test["Survived_f"].astype(int)
test.to_csv('data/vot.csv',index=False)
test[['PassengerId','Survived']].set_index('PassengerId').to_csv('data/sub14.csv')

gbdt.fit(train_data_X_sd,train_data_Y)
test = pd.read_csv('%s/%s' % ('data/', 'test.csv'))
test["Survived_f"] = gbdt.predict(test_data_X_sd)
test['Survived'] = test["Survived_f"].astype(int)
test.to_csv('data/vot.csv',index=False)
test[['PassengerId','Survived']].set_index('PassengerId').to_csv('data/sub8.csv')

xgb_model.fit(train_data_X_sd,train_data_Y)
test = pd.read_csv('%s/%s' % ('data/', 'test.csv'))
test["Survived_f"] = xgb_model.predict(test_data_X_sd)
test['Survived'] = test["Survived_f"].astype(int)
test.to_csv('data/xgb.csv',index=False)
test[['PassengerId','Survived']].set_index('PassengerId').to_csv('data/sub11.csv')
