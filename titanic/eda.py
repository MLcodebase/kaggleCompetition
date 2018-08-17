import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
train_test_ori = train.append(test)
train_test = pd.read_csv('data/pre_proc.csv')

train_age = train[train['Age'].notnull()]

train_age['AgeInt'] = train_age['Age'].apply(lambda x:round(x))
g = sns.FacetGrid(train_age, col='Survived',size=5)
g.map(plt.hist, 'AgeInt', bins=40)
train_age[['AgeInt','Survived']].groupby(['AgeInt']).mean().plot.bar()
train_age[['AgeInt','Survived']].groupby(['AgeInt']).count().plot.bar()
plt.show()
age_level 0-6 7-15 16-47 48-80

train['Cabin_Letter'] = train['Cabin'].apply(lambda x: x[0] if not isinstance(x,float) else x)
train[['Cabin_Letter','Survived']].groupby(['Cabin_Letter']).mean().plot.bar()
train[['Cabin_Letter','Survived']].groupby(['Cabin_Letter']).count().plot.bar()
#eliminate 'G' 'T'
train_test['Cabin_Letter'] = train_test['Cabin'].apply(lambda x: x[0] if not isinstance(x,float) else x)
train_test[train_test['Cabin_Letter'] =='G']

train['Cabin_list'] = train['Cabin'].str.split()
train['Cabin_Num'] = train['Cabin_list'].apply(lambda x: x[0][1:] if not isinstance(x,float) else np.nan)
train[['Cabin_Num','Survived']].groupby(['Cabin_Num']).mean().plot.bar()
train[['Cabin_Num','Survived']].groupby(['Cabin_Num']).count().plot.bar()
#Cabin num sum no less than 5 is a new feature

train[train['Cabin_Num']=='33'][['Cabin','Survived']]

train_test['Cabin_list'] = train_test['Cabin'].str.split()
train_test['Cabin_Num']  = train_test['Cabin_list'].apply(lambda x: x[0][1:] if not isinstance(x,float) else np.nan)
train_test[train_test['Cabin_Num']=='33'][['Cabin','Survived']]


train_data = train_test[:891]
test_data = train_test[891:]
train_data[['Ticket_Num','Survived']].groupby(['Ticket_Num']).mean().plot.bar()
train_data[['Ticket_Num','Survived']].groupby(['Ticket_Num']).count().plot.bar()
train_data['Ticket_Num']

train_data[['Ticket_Num_sum','Survived']].groupby(['Ticket_Num_sum']).mean().plot.bar()
train_data[['Ticket_Num_sum','Survived']].groupby(['Ticket_Num_sum']).count().plot.bar()
train_data[['Ticket_Num_sum']].value_counts()
train_test[train_test['Ticket_Num_sum']==11][['Ticket_Num_sum','Survived','Ticket_Num']]
# ticket num sum one hot eliminate sum no less than 24 and add sum==11


train_data[['Name2_sum','Survived']].groupby(['Name2_sum']).mean().plot.bar()
train_data[['Name2_sum','Survived']].groupby(['Name2_sum']).count().plot.bar()