import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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



cols_to_rename_t1 = {col: f"{col}_t1" for col in train_2_1.columns if col != 'id'}
train_2_1.rename(columns=cols_to_rename_t1, inplace=True)

cols_to_rename_t2 = {col: f"{col}_t2" for col in train_2_2.columns if col != 'id'}
train_2_2.rename(columns=cols_to_rename_t2, inplace=True)

# Merge primary data with secondary data from time period 1
train = pd.merge(train_1, train_2_1, on='id', how='left')

# Merge the result with secondary data from time period 2
train = pd.merge(train, train_2_2, on='id', how='left')

train.to_csv("train.csv", index=False)

# Repeat the same for the test data
# Rename columns in test_2_1
cols_to_rename_t1 = {col: f"{col}_t1" for col in test_2_1.columns if col != 'id'}
test_2_1.rename(columns=cols_to_rename_t1, inplace=True)

# Rename columns in test_2_2
cols_to_rename_t2 = {col: f"{col}_t2" for col in test_2_2.columns if col != 'id'}
test_2_2.rename(columns=cols_to_rename_t2, inplace=True)

# Merge primary test data with secondary data
test = pd.merge(test_1, test_2_1, on='id', how='left')
test = pd.merge(test, test_2_2, on='id', how='left')

print(f'Train shape after merging: {train.shape}')
print(f'Test shape after merging: {test.shape}')

test.to_csv("test.csv", index=False)


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")



# Identify columns with a high percentage of missing values in train
missing_train = train.isnull().mean() * 100
cols_to_drop = missing_train[missing_train > 80].index
train.drop(columns=cols_to_drop, inplace=True)


# Do the same for test data
missing_test = test.isnull().mean() * 100
cols_to_drop_test = missing_test[missing_test > 80].index
test.drop(columns=cols_to_drop_test, inplace=True)

# Align columns between train and test
# common_columns = sorted(list(set(train.columns).intersection(set(test.columns))))
# train = train[common_columns]
# test = test[common_columns]

# Impute missing values
from sklearn.impute import SimpleImputer

numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('label')  # Exclude target variable

imputer = SimpleImputer(strategy='median')

train[numeric_cols] = imputer.fit_transform(train[numeric_cols])
test[numeric_cols] = imputer.transform(test[numeric_cols])

# For categorical variables, fill missing values with mode
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()

for col in categorical_cols:
    mode = train[col].mode()[0]
    train[col].fillna(mode, inplace=True)
    test[col].fillna(mode, inplace=True)



# Using Interquartile Range (IQR) method for outlier detection
Q1 = train[numeric_cols].quantile(0.25)
Q3 = train[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

# Define outlier criterion
is_outlier = (train[numeric_cols] < (Q1 - 1.5 * IQR)) | (train[numeric_cols] > (Q3 + 1.5 * IQR))

# Remove outliers (optional, as removing may not always be beneficial)
# For large datasets, sometimes it's better to cap the outliers
# Here, we'll cap the outliers
train[numeric_cols] = np.where(
    train[numeric_cols] < (Q1 - 1.5 * IQR), Q1 - 1.5 * IQR,
    np.where(train[numeric_cols] > (Q3 + 1.5 * IQR), Q3 + 1.5 * IQR, train[numeric_cols])
)


# Identify the time period columns
t1_cols = [col for col in train.columns if col.endswith('_t1')]
t2_cols = [col for col in train.columns if col.endswith('_t2')]

# Ensure we have matching columns (excluding 'id')
common_add_cols = set([col[:-3] for col in t1_cols]).intersection(set([col[:-3] for col in t2_cols]))

for base_col in common_add_cols:
    col_t1 = base_col + '_t1'
    col_t2 = base_col + '_t2'
    # Create difference and ratio features in train
    train[f'{base_col}_diff'] = train[col_t1] - train[col_t2]
    train[f'{base_col}_ratio'] = train[col_t1] / (train[col_t2] + 1e-5)
    # Create difference and ratio features in test
    test[f'{base_col}_diff'] = test[col_t1] - test[col_t2]
    test[f'{base_col}_ratio'] = test[col_t1] / (test[col_t2] + 1e-5)


from sklearn.preprocessing import LabelEncoder

# Identify categorical columns
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('loan_id')
categorical_cols.remove('id')

# Initialize LabelEncoder
le = LabelEncoder()

for col in categorical_cols:
    # Fit on combined data to ensure consistency
    combined_data = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined_data)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
test[numeric_cols] = scaler.transform(test[numeric_cols])


# Exclude identifiers and target variable
X = train.drop(['loan_id', 'id', 'label'], axis=1)
y = train['label']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

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
    valid_sets=[lgb_train, lgb_eval],
    #early_stopping_rounds=100,
    #verbose_eval=100
)



from sklearn.metrics import roc_auc_score

# Predict on validation data
y_pred_val = model.predict(X_val)

# Calculate AUC-ROC
auc_score = roc_auc_score(y_val, y_pred_val)
print(f'Validation AUC-ROC Score: {auc_score:.4f}')


import matplotlib.pyplot as plt

lgb.plot_importance(model, max_num_features=20)
plt.title('Feature Importance')
plt.show()


from sklearn.model_selection import GridSearchCV

param_grid = {
    'num_leaves': [31, 50, 70],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [-1, 10, 20],
}

lgb_estimator = lgb.LGBMClassifier(
    objective='binary',
    n_estimators=500,
    #early_stopping_rounds=50,
    verbose=1
)

grid = GridSearchCV(estimator=lgb_estimator, param_grid=param_grid,
                    scoring='roc_auc', cv=3, verbose=1, n_jobs=-1)

grid.fit(X, y)
print(f'Best parameters: {grid.best_params_}')
print(f'Best AUC-ROC score: {grid.best_score_}')


params.update(grid.best_params_)



### XGBOOST

import xgboost as xgb
from sklearn.model_selection import train_test_split

# Split data into training and validation sets
X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train_xgb, label=y_train_xgb)
dval = xgb.DMatrix(X_val_xgb, label=y_val_xgb)
dtest = xgb.DMatrix(X_test)

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
evallist = [(dtrain, 'train'), (dval, 'eval')]
xgb_model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evallist,
    early_stopping_rounds=50,
    verbose_eval=100
)

# Evaluate on validation data
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Split data into training and validation sets
X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train_xgb, label=y_train_xgb)
dval = xgb.DMatrix(X_val_xgb, label=y_val_xgb)

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
evallist = [(dtrain, 'train'), (dval, 'eval')]
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
auc_score_xgb = roc_auc_score(y_val_xgb, y_pred_val_xgb)
print(f'XGBoost Validation AUC-ROC Score: {auc_score_xgb:.4f}')


### catboost

from catboost import CatBoostClassifier, Pool

# Create Pools for CatBoost
train_pool = Pool(X_train_xgb, y_train_xgb)
val_pool = Pool(X_val_xgb, y_val_xgb)

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
    eval_set=val_pool,
    use_best_model=True
)

# Evaluate on validation data
y_pred_val_cat = cat_model.predict_proba(X_val_xgb)[:, 1]
auc_score_cat = roc_auc_score(y_val_xgb, y_pred_val_cat)
print(f'CatBoost Validation AUC-ROC Score: {auc_score_cat:.4f}')



### stacking ensemble


from sklearn.model_selection import train_test_split

# Split into training and validation sets
X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


from sklearn.model_selection import KFold

n_splits = 5  # Number of folds
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Arrays to hold out-of-fold predictions and test predictions
oof_train = np.zeros((X_train_full.shape[0], 3))  # 3 base models
oof_test = np.zeros((X_test.shape[0], 3))
oof_test_skf = np.zeros((n_splits, X_test.shape[0], 3))

or i, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
    print(f"Fold {i+1}/{n_splits}")
    X_tr, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
    y_tr, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
    
    # LightGBM
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    oof_train[val_idx, 0] = lgb_model.predict_proba(X_val)[:, 1]
    oof_test_skf[i, :, 0] = lgb_model.predict_proba(X_test)[:, 1]
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(**best_xgb_params, use_label_encoder=False, eval_metric='auc')
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    oof_train[val_idx, 1] = xgb_model.predict_proba(X_val)[:, 1]
    oof_test_skf[i, :, 1] = xgb_model.predict_proba(X_test)[:, 1]
    
    # CatBoost
    cat_model = CatBoostClassifier(**best_cat_params, verbose=False, eval_metric='AUC', early_stopping_rounds=50)
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    oof_train[val_idx, 2] = cat_model.predict_proba(X_val)[:, 1]
    oof_test_skf[i, :, 2] = cat_model.predict_proba(X_test)[:, 1]

# Average test predictions over folds
oof_test = oof_test_skf.mean(axis=0)




### advanced methods to improve accuracy. 


# Example: Multiplying numerical features

import numpy as np
import pandas as pd
from itertools import combinations

# Assuming X and X_test are your DataFrames
num_cols_train = X.select_dtypes(include=[np.number]).columns.tolist()
num_cols_test = X_test.select_dtypes(include=[np.number]).columns.tolist()

# Find the common columns between X and X_test
num_cols = list(set(num_cols_train).intersection(num_cols_test))

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X[num_cols])
X_test_poly = poly.transform(X_test[num_cols])

# Update X and X_test with the new polynomial features
# Use get_feature_names_out if available
try:
    feature_names = poly.get_feature_names_out(num_cols)
except AttributeError:
    # Fallback for older versions of scikit-learn
    feature_names = poly.get_feature_names(num_cols)

X = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
X_test = pd.DataFrame(X_test_poly, columns=feature_names, index=X_test.index)


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42)
X['cluster'] = kmeans.fit_predict(X)
X_test['cluster'] = kmeans.predict(X_test)


import category_encoders as ce

cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Initialize target encoder
target_enc = ce.TargetEncoder(cols=cat_cols)

# Fit on training data
X[cat_cols] = target_enc.fit_transform(X[cat_cols], y)
X_test[cat_cols] = target_enc.transform(X_test[cat_cols])




# Examine class distribution
class_counts = y.value_counts()
print(class_counts)


from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

# Initialize Boruta
boruta = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42)

# Fit Boruta
boruta.fit(X.values, y.values)

# Select features
selected_features = X.columns[boruta.support_].tolist()
print(f'Selected Features: {selected_features}')

# Reduce datasets
X_sel = X[selected_features]
X_test_sel = X_test[selected_features]



import xgboost as xgb
from sklearn.model_selection import train_test_split

# Split data into training and validation sets
X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train_xgb, label=y_train_xgb)
dval = xgb.DMatrix(X_val_xgb, label=y_val_xgb)

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
evallist = [(dtrain, 'train'), (dval, 'eval')]
xgb_model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evallist,
    #early_stopping_rounds=50,
    verbose_eval=100
)

# Evaluate on validation data
y_pred_val_xgb = xgb_model.predict(dval)

from sklearn.metrics import roc_auc_score
auc_score_xgb = roc_auc_score(y_val_xgb, y_pred_val_xgb)
print(f'XGBoost Validation AUC-ROC Score: {auc_score_xgb:.4f}')


### catboost

from catboost import CatBoostClassifier, Pool

# Create Pools for CatBoost
train_pool = Pool(X_train_xgb, y_train_xgb)
val_pool = Pool(X_val_xgb, y_val_xgb)

# Parameters
cat_features = []  # List of categorical feature indices if any

cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric='AUC',
    random_seed=42,
    #early_stopping_rounds=50,
    #verbose=100
)

# Train the model
cat_model.fit(
    train_pool,
    eval_set=val_pool,
    use_best_model=True
)

# Evaluate on validation data
y_pred_val_cat = cat_model.predict_proba(X_val_xgb)[:, 1]
auc_score_cat = roc_auc_score(y_val_xgb, y_pred_val_cat)
print(f'CatBoost Validation AUC-ROC Score: {auc_score_cat:.4f}')




### training neural network 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Define the model
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(X.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='binary_crossentropy', 
              metrics=['AUC'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_auc', 
                               patience=10, 
                               restore_best_weights=True, 
                               mode='max')


# Split data
from sklearn.model_selection import train_test_split

X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit the model
history = model.fit(
    X_train_nn, y_train_nn,
    validation_data=(X_val_nn, y_val_nn),
    epochs=100,
    batch_size=256,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
from sklearn.metrics import roc_auc_score

y_val_pred_nn = model.predict(X_val_nn).ravel()
auc_nn = roc_auc_score(y_val_nn, y_val_pred_nn)
print(f'Neural Network Validation AUC: {auc_nn:.4f}')



###stacking 

from sklearn.model_selection import train_test_split

# Split into training and validation sets
X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.model_selection import KFold

n_splits = 5  # Number of folds
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Arrays to hold out-of-fold predictions and test predictions
oof_train = np.zeros((X_train_full.shape[0], 3))  # 3 base models
oof_test = np.zeros((X_test.shape[0], 3))
oof_test_skf = np.zeros((n_splits, X_test.shape[0], 3))


lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -2,
    'verbose': -1,
}

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

cat_params = {
    'iterations' :1000,
    'learning_rate' :0.05,
    'depth':6,
    'eval_metric':'AUC',
    'random_seed':42,
}

for i, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
    print(f"Fold {i+1}/{n_splits}")
    X_tr, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
    y_tr, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
    
    # LightGBM
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    oof_train[val_idx, 0] = lgb_model.predict_proba(X_val)[:, 1]
    oof_test_skf[i, :, 0] = lgb_model.predict_proba(X_test)[:, 1]
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(**xgb_params, use_label_encoder=False,)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    oof_train[val_idx, 1] = xgb_model.predict_proba(X_val)[:, 1]
    oof_test_skf[i, :, 1] = xgb_model.predict_proba(X_test)[:, 1]
    
    # CatBoost
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    oof_train[val_idx, 2] = cat_model.predict_proba(X_val)[:, 1]
    oof_test_skf[i, :, 2] = cat_model.predict_proba(X_test)[:, 1]

# Average test predictions over folds
oof_test = oof_test_skf.mean(axis=0)


from sklearn.linear_model import LogisticRegression

# Use validation set (from our initial split) to evaluate stacking model
# First, generate base model predictions on X_val_full

val_preds = np.zeros((X_val_full.shape[0], 3))

# LightGBM
lgb_model_full = lgb.LGBMClassifier(**lgb_params)
lgb_model_full.fit(X_train_full, y_train_full)
val_preds[:, 0] = lgb_model_full.predict_proba(X_val_full)[:, 1]
test_preds_lgb = lgb_model_full.predict_proba(X_test)[:, 1]

# XGBoost
xgb_model_full = xgb.XGBClassifier(**xgb_params, use_label_encoder=False)
xgb_model_full.fit(X_train_full, y_train_full)
val_preds[:, 1] = xgb_model_full.predict_proba(X_val_full)[:, 1]
test_preds_xgb = xgb_model_full.predict_proba(X_test)[:, 1]

# CatBoost
cat_model_full = CatBoostClassifier(**cat_params, verbose=False)
cat_model_full.fit(X_train_full, y_train_full)
val_preds[:, 2] = cat_model_full.predict_proba(X_val_full)[:, 1]
test_preds_cat = cat_model_full.predict_proba(X_test)[:, 1]

# Now, train the stacking model on oof_train
stacker = LogisticRegression()
stacker.fit(oof_train, y_train_full)

# Evaluate stacking model on validation set
val_meta_pred = stacker.predict_proba(val_preds)[:, 1]
from sklearn.metrics import roc_auc_score
auc_score_meta = roc_auc_score(y_val_full, val_meta_pred)
print(f'Stacking Ensemble Validation AUC-ROC Score: {auc_score_meta:.4f}')