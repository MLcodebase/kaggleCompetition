import pandas as pd
root_path = 'data'
train_test = pd.read_csv('%s/%s' % (root_path, 'pre_proc.csv'))
missing_age = train_test.drop(['Survived','Cabin_nan','age_nan','corseTitle'],axis=1)
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
missing_age_Y_train_class = pd.cut(missing_age_Y_train,bins=[0,10,18,30,50,100],labels=[1,2,3,4,5])
# use classification instead of regression
from sklearn.ensemble import RandomForestClassifier
rf_c = RandomForestClassifier(random_state=5)
rf_c_scores = cross_validation.cross_val_score(rf_c, missing_age_X_train, missing_age_Y_train_class, cv=5)
from sklearn.ensemble import GradientBoostingClassifier
gbrt_c = GradientBoostingClassifier()
gbrt_c_scores = cross_validation.cross_val_score(gbrt_c, missing_age_X_train, missing_age_Y_train_class, cv=5)
gbrt_c.fit(missing_age_X_train,missing_age_Y_train_class)
train_test.loc[(train_test['Age'].isnull()),  'corseAge'] = gbrt_c.predict(missing_age_X_test)
train_test.loc[(train_test['Age'].notnull()), 'corseAge'] = missing_age_Y_train_class
train_test.to_csv('%s/%s' % (root_path, 'fill_age-cls.csv'),index=False)
