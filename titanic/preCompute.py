import numpy as np
import pandas as pd

root_path = 'data'
train = pd.read_csv('%s/%s' % (root_path, 'train.csv'))
test = pd.read_csv('%s/%s' % (root_path, 'test.csv'))
# train_test df, train with tree model
train_test = train.append(test)

# create new feature,
train_test['SibSp_Parch'] = train_test['SibSp'] + train_test['Parch']
# fill in null data within column Fare
train_test["Fare"].fillna(14.435422,inplace=True)    #fill with mean from pclass and embarked column

# extract title from Name
train_test['fineTitle'] = train_test['Name'].str.extract('.+,(.+)', expand=False).str.extract('^(.+?)\.', expand=False).str.strip()
train_test['corseTitle'] = train_test['fineTitle']
train_test['corseTitle'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer',inplace=True)
train_test['corseTitle'].replace(['Jonkheer', 'Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty',inplace=True)
train_test['corseTitle'].replace(['Mme', 'Ms', 'Mrs'], 'Mrs',inplace=True)
train_test['corseTitle'].replace(['Mlle', 'Miss'], 'Miss',inplace=True)
train_test['corseTitle'].replace(['Mr'], 'Mr',inplace=True)
train_test['corseTitle'].replace(['Master'], 'Master',inplace=True)
# convert into onehot encoding
train_test = pd.get_dummies(train_test,columns=['fineTitle'])
train_test = pd.get_dummies(train_test,columns=['corseTitle'])
# to be continue: extract last name, ticket, age, cabin

train_test = train_test.drop(['Cabin','Survived'],axis=1)
missing_age_train = train_test[train_test['Age'].notnull()]
missing_age_test = train_test[train_test['Age'].isnull()]

# convert categorical feature into one hot encoding, train with factory machine
train_test_one_hot = pd.get_dummies(train_test,columns=['Pclass'])
train_test_one_hot = pd.get_dummies(train_test,columns=["Sex"])
