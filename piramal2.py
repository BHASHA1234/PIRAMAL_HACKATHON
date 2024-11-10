import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool



# Load primary train data
train_1 = pd.read_csv('/Users/krishantsethia/code/handson-ml2/piramal_finance/dataset/train_1.csv')

# Load secondary train data
train_2_1 = pd.read_csv('/Users/krishantsethia/code/handson-ml2/piramal_finance/dataset/test_2_1.csv')
train_2_2 = pd.read_csv('/Users/krishantsethia/code/handson-ml2/piramal_finance/dataset/test_2_2.csv')

# Load primary test data
test_1 = pd.read_csv('/Users/krishantsethia/code/handson-ml2/piramal_finance/dataset/test_1.csv')

# Load secondary test data
test_2_1 = pd.read_csv('/Users/krishantsethia/code/handson-ml2/piramal_finance/dataset/test_2_1.csv')
test_2_2 = pd.read_csv('/Users/krishantsethia/code/handson-ml2/piramal_finance/dataset/train_2_2.csv')



# Rename columns in train_2_1
cols_to_rename_t1 = {col: f"{col}_t1" for col in train_2_1.columns if col != 'id'}
train_2_1.rename(columns=cols_to_rename_t1, inplace=True)

# Rename columns in train_2_2
cols_to_rename_t2 = {col: f"{col}_t2" for col in train_2_2.columns if col != 'id'}
train_2_2.rename(columns=cols_to_rename_t2, inplace=True)

# Merge primary data with secondary data from time period 1
train = pd.merge(train_1, train_2_1, on='id', how='left')

# Merge the result with secondary data from time period 2
train = pd.merge(train, train_2_2, on='id', how='left')

# Do the same for the test data
# Rename columns in test_2_1
cols_to_rename_t1 = {col: f"{col}_t1" for col in test_2_1.columns if col != 'id'}
test_2_1.rename(columns=cols_to_rename_t1, inplace=True)

# Rename columns in test_2_2
cols_to_rename_t2 = {col: f"{col}_t2" for col in test_2_2.columns if col != 'id'}
test_2_2.rename(columns=cols_to_rename_t2, inplace=True)

# Merge primary test data with secondary data
test = pd.merge(test_1, test_2_1, on='id', how='left')
test = pd.merge(test, test_2_2, on='id', how='left')


# Combine for consistent preprocessing
train['dataset'] = 'train'
test['dataset'] = 'test'
combined = pd.concat([train, test], axis=0, ignore_index=True)


# Identify columns with a high percentage of missing values
missing_combined = combined.isnull().mean() * 100
cols_to_drop = missing_combined[missing_combined > 80].index
combined.drop(columns=cols_to_drop, inplace=True)


# Exclude 'loan_id', 'id', 'label', 'dataset'
exclude_cols = ['loan_id', 'id', 'label', 'dataset']

# Categorical columns
categorical_cols = combined.select_dtypes(include=['object']).columns.tolist()
categorical_cols = [col for col in categorical_cols if col not in exclude_cols]


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in categorical_cols:
    combined[col] = combined[col].astype(str)
    combined[col] = le.fit_transform(combined[col])


# Numeric columns
numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in exclude_cols + ['label']]

# Impute numerical columns
from sklearn.impute import SimpleImputer

imputer_num = SimpleImputer(strategy='median')
combined[numeric_cols] = imputer_num.fit_transform(combined[numeric_cols])

# Impute categorical columns
imputer_cat = SimpleImputer(strategy='most_frequent')
combined[categorical_cols] = imputer_cat.fit_transform(combined[categorical_cols])



# Example: Creating time-based difference features
t1_cols = [col for col in combined.columns if col.endswith('_t1')]
t2_cols = [col for col in combined.columns if col.endswith('_t2')]

common_base_cols = list(set([col[:-3] for col in t1_cols]).intersection(set([col[:-3] for col in t2_cols])))

for base_col in common_base_cols:
    col_t1 = base_col + '_t1'
    col_t2 = base_col + '_t2'
    combined[f'{base_col}_diff'] = combined[col_t1] - combined[col_t2]


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
combined[numeric_cols] = scaler.fit_transform(combined[numeric_cols])


# Split back
train = combined[combined['dataset'] == 'train'].drop(['dataset'], axis=1).reset_index(drop=True)
test = combined[combined['dataset'] == 'test'].drop(['dataset', 'label'], axis=1).reset_index(drop=True)



X = train.drop(['loan_id', 'id', 'label'], axis=1)
y = train['label']

X_TEST = test.drop(['loan_id', 'id'], axis=1)



from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42)
X['cluster'] = kmeans.fit_predict(X.drop(columns=['cluster'], errors='ignore'))
X_TEST['cluster'] = kmeans.predict(X_TEST)




from sklearn.model_selection import train_test_split

# First, split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# Then, split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)






import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -2,
    'verbose': -1,
}

# Train the model
model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_test],
    #early_stopping_rounds=100,
    #verbose_eval=100
)



from sklearn.metrics import roc_auc_score

# Predict on validation data
y_pred_val = model.predict(X_val)

# Calculate AUC-ROC
auc_score = roc_auc_score(y_val, y_pred_val)
print(f'Validation AUC-ROC Score: {auc_score:.4f}')



import xgboost as xgb

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
dval = xgb.DMatrix(X_val, label=y_val)


# Parameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

# Train the model
evallist = [(dtrain, 'train'), (dtest, 'test')]
xgb_model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evallist,
    early_stopping_rounds=50,
    verbose_eval=100
)

# Evaluate on validation data
y_pred_val_xgb = xgb_model.predict(dval)

from sklearn.metrics import roc_auc_score
auc_score_xgb = roc_auc_score(y_val, y_pred_val_xgb)
print(f'XGBoost Validation AUC-ROC Score: {auc_score_xgb:.4f}')


### catboost

from catboost import CatBoostClassifier, Pool

# Create Pools for CatBoost
train_pool = Pool(X_train, y_train)
test_pool = Pool(X_test, y_test)
val_pool = Pool(X_val,y_val)

# Parameters
cat_features = []  # List of categorical feature indices if any

cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric='AUC',
    random_seed=42,
    early_stopping_rounds=50,
    verbose=100
)

# Train the model
cat_model.fit(
    train_pool,
    eval_set=test_pool,
    use_best_model=True
)

# Evaluate on validation data
y_pred_val_cat = cat_model.predict_proba(X_val)[:, 1]
auc_score_cat = roc_auc_score(y_val, y_pred_val_cat)
print(f'CatBoost Validation AUC-ROC Score: {auc_score_cat:.4f}')




