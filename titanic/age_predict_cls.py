import pandas as pd
root_path = 'data'
train_test = pd.read_csv('%s/%s' % (root_path, 'pre_proc.csv'))
missing_age = train_test.drop(['Survived','age_nan','Ticket_Num_sum','Ticket_Num','corseTitle'],axis=1)
# process non numeric feature
missing_age = pd.get_dummies(missing_age,columns=['Cabin','Sex','fineTitle','Ticket_Letter','Embarked','Name2_new'])
missing_age_train = missing_age[missing_age['Age'].notnull()]
missing_age_test = missing_age[missing_age['Age'].isnull()]
missing_age_X_train = missing_age_train.drop(['Age'], axis=1)
missing_age_Y_train = missing_age_train['Age']
missing_age_X_test = missing_age_test.drop(['Age'], axis=1)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(missing_age_X_train)
missing_age_X_train = ss.transform(missing_age_X_train)
missing_age_X_test = ss.transform(missing_age_X_test)
from sklearn import cross_validation
missing_age_Y_train_class = pd.cut(missing_age_Y_train,bins=[0,12,20,35,60,100],labels=[1,2,3,4,5])

from sklearn.ensemble import GradientBoostingClassifier
gbrt_c = GradientBoostingClassifier()
param = {"criterion":["mae"],
         "learning_rate":[0.4,0.5,0.6],
         "n_estimators":[95,100,105],
         "min_samples_leaf":[2,3,4],
         "max_depth":[4, 5],
         }
from sklearn.grid_search import GridSearchCV
clf = GridSearchCV(gbrt_c, param,cv=5, n_jobs=-1, verbose=1)
clf.fit(missing_age_X_train,missing_age_Y_train_class)
clf.best_params_
gbrt_c = GradientBoostingClassifier(n_estimators=100,learning_rate=0.6,max_depth=4,min_samples_leaf=3,criterion='mae')
gbrt_c_scores = cross_validation.cross_val_score(gbrt_c, missing_age_X_train, missing_age_Y_train_class, cv=5)
gbrt_c_scores

import xgboost as xgb
xgb_model = xgb.XGBClassifier()
param = {
    "n_estimators":[95,100,105],
    "learning_rate":[0.005,0.01,0.02],
    "max_depth":[4,5]
         }
clf = GridSearchCV(xgb_model, param,cv=5, n_jobs=-1, verbose=1)
clf.fit(missing_age_X_train,missing_age_Y_train_class)
clf.best_params_
#xgb_model = xgb.XGBClassifier(n_estimators=95,learning_rate=0.01,max_depth=4)
xgb_model = xgb.XGBClassifier(n_estimators=100,learning_rate=0.075,max_depth=4)
xg_c_scores = cross_validation.cross_val_score(xgb_model, missing_age_X_train, missing_age_Y_train_class, cv=5)
xg_c_scores

from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
lr = LogisticRegression()
#param = {'C':[0.001,0.01,0.1,1,10], "max_iter":[100,250]}
param = {'C':[0.05,0.075,0.1,0.125,0.15], "max_iter":[50,100,150,200]}
clf = GridSearchCV(lr, param,cv=5, n_jobs=-1, verbose=1)
clf.fit(missing_age_X_train,missing_age_Y_train_class)
clf.best_params_
#lr = LogisticRegression(C=0.1,max_iter=100)
lr = LogisticRegression(C=0.05,max_iter=50)
lr_scores = cross_validation.cross_val_score(lr, missing_age_X_train, missing_age_Y_train_class, cv=5)
lr_scores

# use classification instead of regression
from sklearn.ensemble import RandomForestClassifier
rf_c = RandomForestClassifier()
param = {'random_state':[5,6,8,10,11,12,13,14,15],
         'n_estimators':[40,50],
         'min_samples_leaf': [2],
         'max_depth': [8,9]}
from sklearn.grid_search import GridSearchCV
clf = GridSearchCV(rf_c, param,cv=5, n_jobs=-1, verbose=1)
clf.fit(missing_age_X_train,missing_age_Y_train_class)
clf.best_params_
#rf_c = RandomForestClassifier(oob_score=True,min_samples_leaf=2,n_estimators=40,random_state=10,max_depth=8)/
rf_c = RandomForestClassifier(oob_score=True,min_samples_leaf=2,n_estimators=50,random_state=13,max_depth=9)
rf_c_scores = cross_validation.cross_val_score(rf_c, missing_age_X_train, missing_age_Y_train_class, cv=5)
rf_c_scores

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=0.05,max_iter=50)
import xgboost as xgb
xgb_model = xgb.XGBClassifier(n_estimators=100,learning_rate=0.075,max_depth=4)
from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier(learning_rate=0.6,max_depth=4,n_estimators=100,min_samples_leaf=3,criterion='mae')
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(oob_score=True,min_samples_leaf=2,n_estimators=50,random_state=13,max_depth=9)
vot = VotingClassifier(estimators=[('lr', lr),('gbdt',gbdt),('xgb',xgb_model),('rf',rf)], voting='hard',n_jobs=-1)
from sklearn import cross_validation
vot_scores = cross_validation.cross_val_score(vot, missing_age_X_train, missing_age_Y_train_class, cv=5)
vot_scores
vot.fit(missing_age_X_train,missing_age_Y_train_class)

train_test.loc[(train_test['Age'].isnull()),  'corseAge'] = vot.predict(missing_age_X_test)
train_test.loc[(train_test['Age'].notnull()), 'corseAge'] = missing_age_Y_train_class
train_test.to_csv('%s/%s' % (root_path, 'fill_age-cls.csv'),index=False)
