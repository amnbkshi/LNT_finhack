import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

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

#no need to encode
df['MobileNo_Avl_Flag'] = df['MobileNo_Avl_Flag'].astype('category')
df['Aadhar_flag'] = df['Aadhar_flag'].astype('category')
df['PAN_flag'] = df['PAN_flag'].astype('category')
df['VoterID_flag'] = df['VoterID_flag'].astype('category')
df['Driving_flag'] = df['Driving_flag'].astype('category')
df['Passport_flag'] = df['Passport_flag'].astype('category')
#can encode if computation allows
df['branch_id'] = df['branch_id'].astype('category')
df['manufacturer_id'] = df['manufacturer_id'].astype('category')
df['State_ID'] = df['State_ID'].astype('category')
#encode
df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].astype('category')
df['Employment.Type'] = df['Employment.Type'].astype('category')

le = LabelEncoder()
df['Employment.Type'] = le.fit_transform(df['Employment.Type'])
df['branch_id'] = le.fit_transform(df['branch_id'])
df['manufacturer_id'] = le.fit_transform(df['manufacturer_id'])
df['State_ID'] = le.fit_transform(df['State_ID'])

#new_cols = pd.get_dummies(df[['PERFORM_CNS.SCORE.DESCRIPTION', 'Employment.Type']])
#df = pd.concat([df, new_cols], axis=1)
#df.drop(['Employment.Type', 'PERFORM_CNS.SCORE.DESCRIPTION'], inplace=True, axis=1)

x = df.iloc[:233154,:]
x_test = df.iloc[233154:,:]

cat_indice = [3, 4, 6, 8, 9, 10, 11, 12, 13, 15]
scale_indice = [0, 1, 2, 5, 7, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

sc = StandardScaler()
sc.fit_transform(x.iloc[:, scale_indice].values)
sc.transform(x_test.iloc[:, scale_indice].values)

d_train = lgb.Dataset(x.values, label=y.values, categorical_feature=cat_indice)

params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'auc'
params['sub_feature'] = 0.5
params['num_leaves'] = 120
params['min_data'] = 60
params['max_depth'] = 9
params['scale_pos_weight'] = 4.0 #do_not_change
clf = lgb.train(params, d_train, 1500)

y_pred=clf.predict(x_test.values)

for i in range(len(y_pred)):
    if y_pred[i]>=.5:       
       y_pred[i]=1
    else:  
       y_pred[i]=0

#creating submission file
submission = pd.DataFrame({'UniqueID': unique_id,'loan_default': y_pred})
filename = 'submission.csv'
submission.to_csv(filename,index=False)

