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
# with mean from pclass and embarked column
train_test["Fare"].fillna(14.435422,inplace=True)
# extract title from Name
train_test['fineTitle'] = train_test['Name'].str.extract('.+,(.+)', expand=False).str.extract('^(.+?)\.', expand=False).str.strip()
train_test['corseTitle'] = train_test['fineTitle']
train_test['corseTitle'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer',inplace=True)
train_test['corseTitle'].replace(['Jonkheer', 'Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty',inplace=True)
train_test['corseTitle'].replace(['Mme', 'Ms', 'Mrs'], 'Mrs',inplace=True)
train_test['corseTitle'].replace(['Mlle', 'Miss'], 'Miss',inplace=True)
train_test['corseTitle'].replace(['Mr'], 'Mr',inplace=True)
train_test['corseTitle'].replace(['Master'], 'Master',inplace=True)
# extract last Name
train_test['Name2'] = train_test['Name'].apply(lambda x: x.split('.')[1])
Name2_sum = train_test['Name2'].value_counts().reset_index()
Name2_sum.columns=['Name2','Name2_sum']
train_test = pd.merge(train_test,Name2_sum,how='left',on='Name2')
train_test.loc[train_test['Name2_sum'] == 1 , 'Name2_new'] = 'one'
train_test.loc[train_test['Name2_sum'] > 1 , 'Name2_new'] = train_test['Name2']
del train_test['Name2']
del train_test['Name']
# extract ticket letter
'''
train_test['Ticket_Letter'] = train_test['Ticket'].str.split().str[0]
train_test['Ticket_Letter'] = train_test['Ticket_Letter'].apply(lambda x:np.nan if x.isalnum() else x)
'''
train_test['Ticket_Letter'] = train_test['Ticket'].str.split().str[0]
train_test['Ticket_Letter'] = train_test['Ticket_Letter'].apply(lambda x:np.nan if x.isalnum() else x)
train_test['Ticket_Letter'] = train_test['Ticket_Letter'].str.replace('\.','').str.replace('/','').str.replace('SCParis','SCPARIS')
train_test.drop('Ticket',inplace=True,axis=1)
# process cabin
train_test['Cabin_nan'] = train_test['Cabin'].apply(lambda x:str(x)[0] if pd.notnull(x) else x)
#train_test = pd.get_dummies(train_test,columns=['Cabin_nan'])
train_test.loc[train_test["Cabin"].isnull() ,"Cabin_nan"] = 1
train_test.loc[train_test["Cabin"].notnull() ,"Cabin_nan"] = 0
# create new feature from age info
train_test.loc[train_test["Age"].isnull() ,"age_nan"] = 1
train_test.loc[train_test["Age"].notnull() ,"age_nan"] = 0
# drop unuseful feature
train_test = train_test.drop(['PassengerId'],axis=1)
train_test.to_csv('%s/%s' % (root_path, 'pre_proc.csv'),index=False)
# convert into onehot encoding
train_test = pd.get_dummies(train_test,columns=['age_nan'])
train_test = pd.get_dummies(train_test,columns=['fineTitle'])
train_test = pd.get_dummies(train_test,columns=['corseTitle'])
train_test = pd.get_dummies(train_test,columns=['Name2_new'])
train_test = pd.get_dummies(train_test,columns=['Ticket_Letter'],drop_first=True)
# convert categorical feature into one hot encoding, train with factory machine
train_test = pd.get_dummies(train_test,columns=['Pclass'])
train_test = pd.get_dummies(train_test,columns=["Sex"])
