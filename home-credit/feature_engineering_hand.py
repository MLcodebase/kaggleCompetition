import pandas as pd
import numpy as np
# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns
# Suppress warnings from pandas
import warnings
warnings.filterwarnings('ignore')
import gc
from helpers import *

train = pd.read_csv('data/application_train.csv')
train = convert_types(train)
test = pd.read_csv('data/application_test.csv')
test = convert_types(test)

############## bureau bureau_balance
bureau = pd.read_csv('data/bureau.csv')
bureau = convert_types(bureau, print_info=True)
bureau_counts = agg_categorical(bureau, 'SK_ID_CURR', 'bureau')
bureau_agg = agg_numeric(bureau, 'SK_ID_CURR', 'bureau')

print('bureau aggregation shape: ', bureau_agg.shape)
print('bureau aggregation shape: ', bureau_counts.shape)

train = train.merge(bureau_counts, on ='SK_ID_CURR', how = 'left')
train = train.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')

test = test.merge(bureau_counts, on ='SK_ID_CURR', how = 'left')
test = test.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')

gc.enable()
del bureau_counts,bureau_agg
gc.collect()

print('Training Shape after merge bureau: ', train.shape)
print('Testing Shape  after merge bureau: ', test.shape)

############## bureau bureau_balance
bureau_balance = pd.read_csv('data/bureau_balance.csv')
bureau_balance = convert_types(bureau_balance, print_info=True)
bureau_balance_counts = agg_categorical(bureau_balance, 'SK_ID_BUREAU', 'bureau_balance')
bureau_balance_agg = agg_numeric(bureau_balance, 'SK_ID_BUREAU', 'bureau_balance')

bureau_by_loan = bureau_balance_agg.merge(bureau_balance_counts, right_index = True, left_on = 'SK_ID_BUREAU', how = 'outer')
bureau_by_loan = bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].merge(bureau_by_loan, on = 'SK_ID_BUREAU', how = 'left')
bureau_balance_by_client = agg_numeric(bureau_by_loan.drop(columns = ['SK_ID_BUREAU']), 'SK_ID_CURR', 'client')

print('bureau_balance_by_client shape: ', bureau_balance_by_client.shape)

train = train.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')
test = test.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')

gc.enable()
del bureau,bureau_balance,bureau_balance_counts,bureau_balance_agg,bureau_by_loan, bureau_balance_by_client
gc.collect()

print('Training Shape after merge bureau_balance: ', train.shape)
print('Testing Shape  after merge bureau_balance: ', test.shape)

############## previous_application
previous = pd.read_csv('data/previous_application.csv')
previous = previous.drop()
previous = convert_types(previous, print_info=True)
previous_counts = agg_categorical(previous, 'SK_ID_CURR', 'previous')
previous_agg = agg_numeric(previous, 'SK_ID_CURR', 'previous')

print('Previous aggregation shape: ', previous_agg.shape)
print('Previous aggregation shape: ', previous_counts.shape)

train = train.merge(previous_counts, on ='SK_ID_CURR', how = 'left')
train = train.merge(previous_agg, on = 'SK_ID_CURR', how = 'left')

test = test.merge(previous_counts, on ='SK_ID_CURR', how = 'left')
test = test.merge(previous_agg, on = 'SK_ID_CURR', how = 'left')

gc.enable()
del previous, previous_agg, previous_counts
gc.collect()

print('Training Shape after merge previous_application: ', train.shape)
print('Testing Shape  after merge previous_application: ', test.shape)

############### POS_CASH_balance

cash = pd.read_csv('data/POS_CASH_balance.csv')
cash = convert_types(cash, print_info=True)
cash_by_client = aggregate_client(cash, group_vars = ['SK_ID_PREV', 'SK_ID_CURR'], df_names = ['cash', 'client'])

print('Cash by Client Shape: ', cash_by_client.shape)
train = train.merge(cash_by_client, on = 'SK_ID_CURR', how = 'left')
test = test.merge(cash_by_client, on = 'SK_ID_CURR', how = 'left')

gc.enable()
del cash, cash_by_client
gc.collect()

print('Training Shape after merge POS_CASH_balance: ', train.shape)
print('Testing Shape  after merge POS_CASH_balance: ', test.shape)

############### credit_card_balance
credit = pd.read_csv('data/credit_card_balance.csv')
credit = convert_types(credit, print_info = True)
credit_by_client = aggregate_client(credit, group_vars = ['SK_ID_PREV', 'SK_ID_CURR'], df_names = ['credit', 'client'])
print('Credit by client shape: ', credit_by_client.shape)
train = train.merge(credit_by_client, on = 'SK_ID_CURR', how = 'left')
test = test.merge(credit_by_client, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit, credit_by_client
gc.collect()

print('Training Shape after merge credit_card_balance: ', train.shape)
print('Testing Shape  after merge credit_card_balance: ', test.shape)

############## installments_payments
installments = pd.read_csv('data/installments_payments.csv')
installments = convert_types(installments, print_info = True)
installments_by_client = aggregate_client(installments, group_vars = ['SK_ID_PREV', 'SK_ID_CURR'], df_names = ['installments', 'client'])
print('installments by client shape: ', installments_by_client.shape)

train = train.merge(installments_by_client, on = 'SK_ID_CURR', how = 'left')
test = test.merge(installments_by_client, on = 'SK_ID_CURR', how = 'left')

gc.enable()
del installments, installments_by_client
gc.collect()

train, test = remove_missing_columns(train, test)

print('Final Training Shape: ', train.shape)
print('Final Testing Shape: ', test.shape)

train_labels = train['TARGET']
# Align the dataframes, this will remove the 'TARGET' column
train, test = train.align(test, join = 'inner', axis = 1)
train['TARGET'] = train_labels

print('Training Data Shape after align: ', train.shape)
print('Testing Data Shape  after align: ', test.shape)

submission, fi, metrics = model(train, test)
