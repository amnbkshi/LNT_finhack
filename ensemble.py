import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import catboost as cb
from xgboost import XGBClassifier

train = pd.read_csv('train_aox2Jxw/train.csv', parse_dates=['Date.of.Birth', 'DisbursalDate'])
test = pd.read_csv('test_bqCt9Pv.csv', parse_dates=['Date.of.Birth', 'DisbursalDate'])

unique_id = test['UniqueID']
y = train.iloc[:, -1]
train.drop('loan_default', inplace=True, axis = 1)

df = pd.concat([train, test], axis = 0)

df.drop(['UniqueID', 'Employee_code_ID', 'supplier_id', 'Current_pincode_ID'], inplace=True, axis=1)
df['Employment.Type'].fillna('temp', inplace=True)

df['CREDIT.HISTORY.LENGTH'] = df['CREDIT.HISTORY.LENGTH'].apply(lambda x: int(x[0]))
df['AVERAGE.ACCT.AGE'] = df['AVERAGE.ACCT.AGE'].apply(lambda x: int(x[0]))

df['Date.of.Birth'] = np.abs(2019 - df['Date.of.Birth'].dt.year)
df['DisbursalDate'] = df['DisbursalDate'].dt.month

df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].replace('No Bureau History Available', 0)
df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: Sufficient History Not Available', 0)
df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: Not Enough Info available on the customer', 0)
df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: No Activity seen on the customer (Inactive)',0)
df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: No Updates available in last 36 months', 0)
df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: Only a Guarantor', 0)
df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: More than 50 active Accounts found',0)
df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].replace('M-Very High Risk', 1)
df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].replace('L-Very High Risk', 1)
df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].replace('K-High Risk', 2)
df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].replace('J-High Risk', 2)
df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].replace('I-Medium Risk', 3)
df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].replace('H-Medium Risk', 3)
df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].replace('G-Low Risk', 4)
df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].replace('F-Low Risk', 4)
df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].replace('E-Low Risk', 4)
df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].replace('D-Very Low Risk', 5)
df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].replace('C-Very Low Risk', 5)
df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].replace('B-Very Low Risk', 5)
df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].replace('A-Very Low Risk', 5)

le = LabelEncoder()
df['Employment.Type'] = le.fit_transform(df['Employment.Type'])
df['branch_id'] = le.fit_transform(df['branch_id'])
df['manufacturer_id'] = le.fit_transform(df['manufacturer_id'])
df['State_ID'] = le.fit_transform(df['State_ID'])

x = df.iloc[:233154,:]
x_test = df.iloc[233154:,:]

cat_indice = [3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 16]

#Light GBM
d_train = lgb.Dataset(x.values, label=y.values, categorical_feature=cat_indice)

params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'auc'
params['sub_feature'] = 0.5
params['num_leaves'] = 200
params['min_data'] = 100
params['max_depth'] = 9
params['scale_pos_weight'] = 4.0 #do_not_change
clf1 = lgb.train(params, d_train, 2000)

y_pred_lgbm=clf1.predict(x_test.values)



#CatBoost
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)

eval_dataset = cb.Pool(x_valid.values, y_valid.values, cat_features=cat_indice)
clf2 = cb.CatBoostClassifier(iterations=1000, learning_rate=0.01, depth=6, l2_leaf_reg=3, loss_function='Logloss', border_count=32,
							  use_best_model=True, eval_metric='AUC', scale_pos_weight=4.0)
clf2.fit(x_train, y_train, eval_set=eval_dataset, cat_features = cat_indice)

y_pred_cb = clf2.predict(x_test, prediction_type='Probability')



#XGBoost
clf3 = XGBClassifier(learning_rate = 0.01, min_child_weight=10,max_depth=8, scale_pos_weight = 4.0, n_estimators = 500)
clf3.fit(x, y)

y_pred_xgb = clf3.predict_proba(x_test)



y_pred = (y_pred_lgbm + y_pred_cb[:,1] + y_pred_xgb[:,1]) / 3

for i in range(len(y_pred)):
    if y_pred[i]>=.5:       
       y_pred[i]=1
    else:  
       y_pred[i]=0

#creating submission file
submission = pd.DataFrame({'UniqueID': unique_id,'loan_default': y_pred})
filename = 'submission.csv'
submission.to_csv(filename,index=False)
