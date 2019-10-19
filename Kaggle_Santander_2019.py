import numpy as np
import pandas as pd
# from numba import jit
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn import svm, datasets
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
import warnings


# digits = datasets.load_digits()
# iris = datasets.load_iris()
#
# print(iris.data)
# print(iris.target)

# Testing GridSearch with small dataset
# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# svc = svm.SVC(gamma="scale")
# clf = GridSearchCV(svc, parameters, cv=5)
# clf.fit(iris.data, iris.target)

# Input data
warnings.filterwarnings('ignore')
print(os.listdir("D:\\Kaggle\\Santander\\santander-customer-transaction-prediction"))

train_df = pd.read_csv('D:\\Kaggle\\Santander\\santander-customer-transaction-prediction\\train_fixed.csv')
test_df = pd.read_csv('D:\\Kaggle\\Santander\\santander-customer-transaction-prediction\\test.csv')

test_df.shape, train_df.shape

train_df.head(), test_df.head()

train_df.isnull().values.any()

test_df.isnull().values.any()

# train_df1 = train_df.sample(frac=0.5, replace=True, random_state=1)
# train_df2 = train_df.sample(frac=0.5, replace=True, random_state=100)

train_df_target = train_df['target']
train_df_features = train_df.loc[:, train_df.columns != 'target']

X_train, X_test, y_train, y_test = train_test_split(
    train_df_features, train_df_target, test_size=0.5, random_state=0)

# Plot target variable
sns.set_style('whitegrid')
sns.countplot(train_df['target'])
sns.set_style('whitegrid')

features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
target = train_df['target']

# Shuffling augmentation
def augment(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y

# Parameters
param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.335,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.041,
    'learning_rate': 0.083,
    'max_depth': -1,
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary',
    'verbosity': -1
}

# Parameter range for LightGBM when doing grid search
param_gridsearch = {
    'learning_rate': [0.01, 0.05, 0.10],
    'min_data_in_leaf': [20, 40, 60, 80, 100],
    'min_sum_hessian_in_leaf': [10.0, 20.0],
    'num_leaves': [2, 20],
    'reg_alpha': [0.1, 0.5],
    'lambda_l1': [0, 1, 1.5],
    'lambda_l2': [0, 1]
}

#Parameter for Gridsearch
fit_params = {
    "early_stopping_rounds":1,
    "eval_set" : [[X_test, y_test]]
}
#kfold = 15. Creating folds for CV
#folds = StratifiedKFold(n_splits=kfold, shuffle=False, random_state=44000)
num_folds = 11
features = [c for c in train_df.columns if c not in ['ID_code', 'target']]

folds = KFold(n_splits=num_folds, random_state=44000)
oof = np.zeros(len(train_df))
getVal = np.zeros(len(train_df))
predictions = np.zeros(len(target))
feature_importance_df = pd.DataFrame()


print('Light GBM Model Grid Search')

lgb_estimator = lgb.LGBMRegressor()
# lgb_estimator = lgb.LGBMClassifier( bagging_freq =5,
#     bagging_fraction = 0.335,
#     boost= "gbdt",
#     feature_fraction = 0.041,
#     learning_rate =0.083,
#     max_depth =-1,
#     metric="auc",
#     min_data_in_leaf= 80,
#     min_sum_hessian_in_leaf= 10.0,
#     num_leaves= 13,
#     num_threads= 8,
#     tree_learner= "serial",
#     objective= "binary",
#     verbosity= -1)

gsearch = GridSearchCV(estimator=lgb_estimator, param_grid=param_gridsearch, cv=folds, fit_params=fit_params, scoring="roc_auc")
# gsearch = GridSearchCV(estimator=lgb_estimator, param_grid=param_gridsearch, cv=folds, scoring="roc_auc")

lgb_model = gsearch.fit(X=train_df_features, y=train_df_target)
print(lgb_model.best_params_, lgb_model.best_estimator_, lgb_model.best_params_, sep=" ")
print('Light GBM Model')

# Build LightGBM with the best parameters (currently manually entered)
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
    X_train, y_train = train_df.iloc[trn_idx][features], target.iloc[trn_idx]
    X_valid, y_valid = train_df.iloc[val_idx][features], target.iloc[val_idx]

    X_tr, y_tr = augment(X_train.values, y_train.values)
    X_tr = pd.DataFrame(X_tr)

    print("Fold idx:{}".format(fold_ + 1))
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_valid, label=y_valid)
    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])

    clf = lgb.train(param, trn_data, 1000000, valid_sets=[trn_data, val_data], verbose_eval=5000,
                    early_stopping_rounds=3000)
    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)
    getVal[val_idx] += clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration) / folds.n_splits
    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

print("\n >> CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

# Plot feature importance
cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)
best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14,26))
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('LightGBM Features (averaged over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')

# Generate output scoring file
submission = pd.DataFrame({"ID_code": test_df.ID_code.values})
submission["target"] = predictions
submission.to_csv("D:\\Kaggle\\Santander\\santander-customer-transaction-prediction\\predictions\\submission.csv", index=False)