import pandas as pd
root_path = 'data'
train_test = pd.read_csv('%s/%s' % (root_path, 'pre_proc.csv'))
missing_age = train_test.drop(['Survived','Cabin','age_nan'],axis=1)
# process non numeric feature
#missing_age = pd.get_dummies(missing_age,columns=['Sex','Embarked','fineTitle','Name2_new','Ticket_Letter'])
#missing_age = missing_age.drop(['corseTitle'],axis=1)
missing_age = pd.get_dummies(missing_age,columns=['Sex','Embarked','fineTitle','Ticket_Letter'])
missing_age = missing_age.drop(['corseTitle','Name2_new'],axis=1)
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
from sklearn import linear_model
lin = linear_model.BayesianRidge()
lin_scores = cross_validation.cross_val_score(lin, missing_age_X_train, missing_age_Y_train, cv=5, scoring='mean_absolute_error')
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=5)
rf_scores = cross_validation.cross_val_score(rf, missing_age_X_train, missing_age_Y_train, cv=5, scoring='mean_absolute_error')
from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor()
gbrt_scores = cross_validation.cross_val_score(gbrt, missing_age_X_train, missing_age_Y_train, cv=5, scoring='mean_absolute_error')
gbrt.fit(missing_age_X_train,missing_age_Y_train)
train_test.loc[(train_test['Age'].isnull()), 'Age'] = gbrt.predict(missing_age_X_test)
train_test['corseAge'] = pd.cut(train_test['Age'], bins=[0,10,18,30,50,100],labels=[1,2,3,4,5])
train_test.to_csv('%s/%s' % (root_path, 'fill_age.csv'),index=False)