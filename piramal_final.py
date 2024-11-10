import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


missing_train = train.isnull().mean() * 100
cols_to_drop = missing_train[missing_train > 80].index
train.drop(columns=cols_to_drop, inplace=True)


# Do the same for test data
missing_test = test.isnull().mean() * 100
cols_to_drop_test = missing_test[missing_test > 80].index
test.drop(columns=cols_to_drop_test, inplace=True)

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


X = train.drop(['loan_id', 'id', 'label'], axis=1)
y = train['label']

X_TEST = test.drop(['loan_id', 'id'], axis=1)



from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42)
X['cluster'] = kmeans.fit_predict(X)
X_TEST['cluster'] = kmeans.predict(X_test)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



###lightgbm


import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

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