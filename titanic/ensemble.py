import pandas as pd
import numpy as np
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
X = train_data_X_sd
X_predict = test_data_X_sd
y = train_data_Y
'''模型融合中使用到的各个单模型'''
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
clfs = [LogisticRegression(C=0.1,max_iter=100),
        xgb.XGBClassifier(max_depth=6,n_estimators=100,num_round = 5),
        RandomForestClassifier(n_estimators=100,max_depth=6,oob_score=True),
        GradientBoostingClassifier(learning_rate=0.3,max_depth=6,n_estimators=100)]
# 创建n_folds
from sklearn.cross_validation import StratifiedKFold
n_folds = 5
skf = list(StratifiedKFold(y, n_folds))
# 创建零矩阵
dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_predict.shape[0], len(clfs)))
# 建立模型
for j, clf in enumerate(clfs):
    dataset_blend_test_j = np.zeros((X_predict.shape[0], len(skf)))
    for i, (train, test) in enumerate(skf):
        X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
        clf.fit(X_train, y_train)
        y_submission = clf.predict_proba(X_test)[:, 1]
        dataset_blend_train[test, j] = y_submission
        dataset_blend_test_j[:, i] = clf.predict_proba(X_predict)[:, 1]
    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
# 用建立第二层模型
clf2 = LogisticRegression(C=0.1,max_iter=100)
from sklearn import cross_validation
clf2_scores = cross_validation.cross_val_score(clf2,dataset_blend_train,y,cv=5)
clf2.fit(dataset_blend_train, y)
y_submission = clf2.predict_proba(dataset_blend_test)[:, 1]
test = pd.read_csv("data/test.csv")
test["Survived_f"] = clf2.predict(dataset_blend_test)
test['Survived']=test["Survived_f"].astype(int)
test[['PassengerId','Survived']].set_index('PassengerId').to_csv('sub3.csv')