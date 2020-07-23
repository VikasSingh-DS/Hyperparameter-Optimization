import pandas as pd
import numpy as np

from sklearn import model_selection
import xgboost


df = pd.read_csv("datasets_train.csv")
X = df.drop("price_range", axis=1).values  # input feature
y = df.price_range.values  # target feature

# Define the model
xgb_opt = xgboost.XGBClassifier(tree_method="gpu_hist")


# Define hyperparameters of Xgboost
params = {
    "eta": [0.25],
    "max_depth": [5, 10],
    "min_child_weight": [1, 10],
    "subsample": [0.7, 0.05],
    "gamma": [0.5, 0.05],
    "colsample_bytree": [0.7, 0.05],
    "alpha": [10, 1],
    "lambda": [1, 0.1],
}


# Construct grid search object with 3 fold cross validation
kfold = model_selection.StratifiedKFold(n_splits=5, shuffle=True)
gridcv = model_selection.GridSearchCV(
    estimator=xgb_opt,
    param_grid=params,
    cv=kfold,
    verbose=1,
    scoring="accuracy",
    n_jobs=-1,
)

# Fitting and Performing Turning
gridcv.fit(X, y)

# if you want to know best parametes and best score
print(gridcv.best_score_)
print(gridcv.best_params_)

