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

train['cur_app_term'] = train['AMT_CREDIT'] / train['AMT_ANNUITY']
train['cur_loan_rate'] = train['AMT_CREDIT'] / train['AMT_INCOME_TOTAL']
train['cur_buy_pressure'] = train['AMT_GOODS_PRICE'] / train['AMT_INCOME_TOTAL']
train['cur_loan_good_rate'] = train['AMT_CREDIT'] / train['AMT_GOODS_PRICE']

test['cur_app_term'] = test['AMT_CREDIT'] / test['AMT_ANNUITY']
test['cur_loan_rate'] = test['AMT_CREDIT'] / test['AMT_INCOME_TOTAL']
test['cur_buy_pressure'] = test['AMT_GOODS_PRICE'] / test['AMT_INCOME_TOTAL']
test['cur_loan_good_rate'] = test['AMT_CREDIT'] / test['AMT_GOODS_PRICE']

train_income = train[['SK_ID_CURR','AMT_INCOME_TOTAL']]
test_income  = test[['SK_ID_CURR','AMT_INCOME_TOTAL']]
income = train_income.append(test_income,ignore_index=True)
############## bureau bureau_balance
bureau = pd.read_csv('data/bureau.csv')
bureau = convert_types(bureau, print_info=True)
bureau_loan_count = bureau['SK_ID_CURR'].value_counts().reset_index(name='SK_ID_CURR').rename({'index':'SK_ID_CURR','SK_ID_CURR':'bureau_loan_count'},axis=1)
train = train.merge(bureau_loan_count, on ='SK_ID_CURR', how = 'left')
test = test.merge(bureau_loan_count, on ='SK_ID_CURR', how = 'left')

drop_columns =['DAYS_CREDIT','DAYS_CREDIT_ENDDATE','DAYS_ENDDATE_FACT']
bureau = bureau.drop(columns=drop_columns,axis=1)
bureau = bureau.merge(income,how='left',on='SK_ID_CURR')
bureau['loan_rate'] = bureau['AMT_ANNUITY'] / bureau['AMT_INCOME_TOTAL']
bureau['bureau_app_term'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_ANNUITY']

bureau = bureau.drop(columns=['AMT_INCOME_TOTAL'],axis=1)
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
drop_columns = ['MONTHS_BALANCE']
bureau_balance = bureau_balance.drop(columns=drop_columns,axis=1)
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
previous = convert_types(previous, print_info=True)
drop_columns = ['FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY','DAYS_DECISION','NAME_CLIENT_TYPE','DAYS_FIRST_DRAWING','DAYS_TERMINATION']
previous = previous.drop(columns=drop_columns,axis=1)

previous['previous_app_gap'] = previous['AMT_CREDIT'] / previous['AMT_CREDIT']
previous['previous_app_term'] = previous['AMT_CREDIT'] / previous['AMT_ANNUITY']
previous['downpay_pressure'] = previous['AMT_DOWN_PAYMENT'] / previous['AMT_CREDIT']
previous = previous.merge(income,how='left',on='SK_ID_CURR')
previous['downpay_pressure'] = previous['AMT_DOWN_PAYMENT'] / previous['AMT_INCOME_TOTAL']
previous['previous_buy_pressure'] = previous['AMT_GOODS_PRICE'] / previous['AMT_INCOME_TOTAL']
previous['previous_loan_good_rate'] = previous['AMT_CREDIT'] / previous['AMT_GOODS_PRICE']

previous_loan_count = previous['SK_ID_CURR'].value_counts().reset_index(name='SK_ID_CURR').rename({'index':'SK_ID_CURR','SK_ID_CURR':'previous_loan_count'},axis=1)
train = train.merge(previous_loan_count, on ='SK_ID_CURR', how = 'left')
test = test.merge(previous_loan_count, on ='SK_ID_CURR', how = 'left')

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
drop_columns = ['MONTHS_BALANCE','CNT_INSTALMENT','CNT_INSTALMENT_FUTURE','NAME_CONTRACT_STATUS']
cash = cash.drop(columns=drop_columns,axis=1)

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

drop_columns = ['MONTHS_BALANCE']
credit = credit.drop(columns=drop_columns,axis=1)
credit['card_draw_pay_rate'] = credit['AMT_DRAWINGS_CURRENT'] / credit['AMT_PAYMENT_CURRENT']
credit['card_int_pay_rate'] = credit['AMT_INST_MIN_REGULARITY'] / credit['AMT_PAYMENT_TOTAL_CURRENT']
credit['card_inst_pay_rate'] = credit['AMT_INST_MIN_REGULARITY'] / credit['AMT_PAYMENT_CURRENT']

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
drop_columns = ['DAYS_INSTALMENT','DAYS_ENTRY_PAYMENT']
installments = installments.drop(columns=drop_columns,axis=1)
installments['payment_rate'] = installments['AMT_PAYMENT'] / installments['AMT_INSTALMENT']
install_ver_chg = installments[['SK_ID_CURR','SK_ID_PREV','NUM_INSTALMENT_VERSION']]
install_ver_chg = install_ver_chg.groupby(['SK_ID_CURR','SK_ID_PREV']).nunique()
install_ver_chg = install_ver_chg.drop(columns= ['SK_ID_CURR','SK_ID_PREV'],axis=1).reset_index()
install_ver_chg['chg_cnt'] = install_ver_chg['NUM_INSTALMENT_VERSION'] - 1
install_ver_chg = install_ver_chg.drop(columns=['SK_ID_PREV','NUM_INSTALMENT_VERSION'],axis=1)
train = train.merge(install_ver_chg, on = 'SK_ID_CURR', how = 'left')
test = test.merge(install_ver_chg, on = 'SK_ID_CURR', how = 'left')
installments['pay_day_dif'] = installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT']

def f(x):
  if x == np.nan:
    return x
  else:
    if x>0:
	  return x
    else:
	  return 0
installments['has_default_day'] = installments['pay_day_dif'].apply(f)
installments['payment_rate'] = installments['AMT_PAYMENT'] / installments['AMT_INSTALMENT']

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
