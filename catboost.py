import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import catboost as cb

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

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)

eval_dataset = cb.Pool(x_valid.values, y_valid.values, cat_features=cat_indice)


model = cb.CatBoostClassifier(iterations=600, learning_rate=0.1, depth=6, l2_leaf_reg=3, loss_function='Logloss', border_count=32,
							  use_best_model=True, eval_metric='AUC', scale_pos_weight=3.6)
							  
model.fit(x_train, y_train, eval_set=eval_dataset, cat_features = cat_indice)
print(model.get_best_iteration())
print(model.get_best_score())

y_pred = model.predict(x_test)

#creating submission file
submission = pd.DataFrame({'UniqueID': unique_id,'loan_default': y_pred})
filename = 'submission.csv'
submission.to_csv(filename,index=False)

