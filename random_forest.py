import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

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

new_cols = pd.get_dummies(df[['PERFORM_CNS.SCORE.DESCRIPTION', 'Employment.Type']])
df = pd.concat([df, new_cols], axis=1)
df.drop(['Employment.Type', 'PERFORM_CNS.SCORE.DESCRIPTION'], inplace=True, axis=1)

x = df.iloc[:233154,:]
x_test = df.iloc[233154:,:]

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = [int(x) for x in np.linspace(start = 1, stop = 40, num = 10)]
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
class_weight = [{0:1,1:3},{0:1,1:4},{0:1,1:5}]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'class_weight': class_weight}

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42)

rf_random.fit(x, y)

print(rf_random.best_params_)

param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 5, verbose = 2)

grid_search.fit(x, y)

print(grid_search.best_params_)



y_pred = model_rf.predict(x_test)

#creating submission file
submission = pd.DataFrame({'UniqueID': unique_id,'loan_default': y_pred})
filename = 'submission.csv'
submission.to_csv(filename,index=False)

