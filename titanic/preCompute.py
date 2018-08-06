import numpy as np
import pandas as pd

root_path = 'data'
train = pd.read_csv('%s/%s' % (root_path, 'train.csv'))
test = pd.read_csv('%s/%s' % (root_path, 'test.csv'))
total = train.append(test)

total["Fare"].fillna(14.435422,inplace=True)    #fill with mean from pclass and embarked column

total_age = total.drop(['Cabin','Survived'],axis=1)

missing_age_train = total_age[total_age['Age'].notnull()]
missing_age_test = total_age[total_age['Age'].isnull()]

missing_age_train.info()