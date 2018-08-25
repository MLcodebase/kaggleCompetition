import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns

app_train = pd.read_csv('data/application_train.csv')
app_test  = pd.read_csv('data/application_train.csv')

app_train[['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','NAME_INCOME_TYPE']]

columns_drop = ['SK_ID_CURR',]

columns_categorical_compose = [ ('FLAG_OWN_CAR','FLAG_OWN_REALTY') ] 
 
app_train['TARGET'].corr( app_train['AMT_GOODS_PRICE'] / app_train['AMT_INCOME_TOTAL'] )

app_train['TARGET'].corr( app_train['AMT_CREDIT'] / app_train['AMT_ANNUITY'] )

app_train['loan_term'] = ( app_train['AMT_CREDIT'] / app_train['AMT_ANNUITY'] )

app_train['loan_term_int'] = app_train['loan_term'].apply(round)

app_train[['loan_term_int','TARGET']].groupby('loan_term_int').mean().plot.bar()

plt.hist(app_train['loan_term_int'], edgecolor = 'k', bins = 20)

poor = app_train.loc[ (app_train['FLAG_OWN_CAR']=='Y') & ( app_train['FLAG_OWN_REALTY']=='Y'),['TARGET','AMT_GOODS_PRICE','AMT_INCOME_TOTAL']]
poor['TARGET'].corr( poor['AMT_GOODS_PRICE'] - poor['AMT_INCOME_TOTAL'] )
( poor['AMT_GOODS_PRICE'] - poor['AMT_INCOME_TOTAL'] ).corr(poor['TARGET'])
poor['dif'] = poor['AMT_GOODS_PRICE'] - poor['AMT_INCOME_TOTAL']
sns.kdeplot(poor.loc[poor['TARGET'] == 0, 'dif'] , label = 'target == 0')
sns.kdeplot(poor.loc[poor['TARGET'] == 1, 'dif'] , label = 'target == 1')


featrue_goups = [ 'family', 'economy', 'career', 'event', 'geo']

app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram')
plt.hist(app_train['DAYS_BIRTH'] / -365, edgecolor = 'k', bins = 25, range = (0,40))
plt.hist(app_train['SK_ID_CURR'], edgecolor = 'k', bins = 20)

sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'target == 0')
sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')



def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()    
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")
    # Return the dataframe with missing information
    return mis_val_table_ren_columns

missing_values_table(app_train)
missing_values_table(app_test)

app_train.dtypes.value_counts()
app_train.dtypes.value_counts()
app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)

# Create a label encoder object
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in app_train:
    if app_train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(app_train[col].unique())) <= 2:
            # Train on the training data
            le.fit(app_train[col])
            # Transform both training and testing data
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])    
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)

# one-hot encoding of categorical variables
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)

train_labels = app_train['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)

# Add the target back in
app_train['TARGET'] = train_labels

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)

# https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction

train_labels = app_train['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)

# Add the target back in
app_train['TARGET'] = train_labels

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)
