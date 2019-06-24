import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier

train = pd.read_csv('train_aox2Jxw/train.csv', parse_dates=['Date.of.Birth', 'DisbursalDate'])
test = pd.read_csv('test_bqCt9Pv.csv', parse_dates=['Date.of.Birth', 'DisbursalDate'])

unique_id = test['UniqueID']
y = train.iloc[:, -1]
train.drop('loan_default', inplace=True, axis = 1)

df = pd.concat([train, test], axis = 0)

df.drop(['UniqueID', 'Employee_code_ID', 'supplier_id', 'Current_pincode_ID'], inplace=True, axis=1)
df['Employment.Type'].fillna('temp', inplace=True)


df['CREDIT.HISTORY.LENGTH'] = df['CREDIT.HISTORY.LENGTH'].apply(lambda x: x[0])
df['AVERAGE.ACCT.AGE'] = df['AVERAGE.ACCT.AGE'].apply(lambda x: x[0])

df['Date.of.Birth'] = np.abs(2019 - df['Date.of.Birth'].dt.year)
df['DisbursalDate'] = df['DisbursalDate'].dt.month

cat_indice = np.array([3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 16])

x = df.iloc[:233154,:]
x_test = df.iloc[233154:,:]


bayes_cv_tuner = BayesSearchCV(estimator = CatBoostClassifier(),
                                search_spaces = {
                                'learning_rate':(0.01,2.0, 'uniform'),
                                'depth':(2,10),
                                'l2_leaf_reg':(1, 100),
                                'bagging_temperature':(1e-9, 1000, 'log-uniform'),
                                'border_count':(1,100),
                                'rsm':(0.01, 1.0, 'uniform'),
                                'random_strength':(1e-9, 10, 'log-uniform'),
                                'scale_pos_weight':(2.0, 5.0, 'uniform'),
                                },
                                scoring = 'roc_auc',
                                cv = StratifiedKFold(
                                n_splits=2,
                                shuffle=True,
                                random_state=72
                                ),
                                n_jobs = 1,
                                n_iter = 100,
                                refit = True,
                                random_state = 72
                               )


