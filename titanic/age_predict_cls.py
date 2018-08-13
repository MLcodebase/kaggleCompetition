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
#gbrt_c = GradientBoostingClassifier(learning_rate=0.1,max_depth=6,n_estimators=200,min_samples_leaf=2)
gbrt_c = GradientBoostingClassifier()
param = { "max_depth":[5,6,7],
         "n_estimators":[50,75,100,120],"criterion":['friedman_mse','mae'],
         "min_samples_leaf":[2,3,4]}
from sklearn.grid_search import GridSearchCV
clf = GridSearchCV(gbrt_c, param,cv=4, n_jobs=-1, verbose=1)
clf.fit(missing_age_X_train,missing_age_Y_train_class)
clf.best_params_
gbrt_c = GradientBoostingClassifier(n_estimators=50,learning_rate=0.1,max_depth=6,min_samples_leaf=4)
gbrt_c_scores = cross_validation.cross_val_score(gbrt_c, missing_age_X_train, missing_age_Y_train_class, cv=5)
gbrt_c.fit(missing_age_X_train,missing_age_Y_train_class)

import xgboost as xgb
xgb_model = xgb.XGBClassifier()
xg_c_scores = cross_validation.cross_val_score(xgb_model, missing_age_X_train, missing_age_Y_train_class, cv=5)
xgb_model.fit(missing_age_X_train,missing_age_Y_train_class)

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

train_test.loc[(train_test['Age'].isnull()),  'corseAge'] = xgb_model.predict(missing_age_X_test)
train_test.loc[(train_test['Age'].notnull()), 'corseAge'] = missing_age_Y_train_class
train_test.to_csv('%s/%s' % (root_path, 'fill_age-cls.csv'),index=False)

# use classification instead of regression
from sklearn.ensemble import RandomForestClassifier
rf_c = RandomForestClassifier(random_state=5)
rf_c_scores = cross_validation.cross_val_score(rf_c, missing_age_X_train, missing_age_Y_train_class, cv=5)
