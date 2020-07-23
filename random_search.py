"""Random search is faster than Grid search"""

import pandas as pd
import numpy as np
from scipy.stats import uniform, randint


from sklearn import model_selection
import xgboost


df = pd.read_csv("datasets_train.csv")
X = df.drop("price_range", axis=1).values  # input feature
y = df.price_range.values  # target feature

# Define the model
xgb_opt = xgboost.XGBClassifier(tree_method="gpu_hist")

# Define hyperparameters of Xgboost
params = {
    "n_estimators": randint(1000, 2000),
    "learning_rate": uniform(0.01, 0.06),
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
regcv = model_selection.RandomizedSearchCV(
    estimator=xgb_opt,
    param_distributions=params,
    cv=kfold,
    verbose=1,
    scoring="accuracy",
    n_jobs=-1,
)

# Fitting and Performing Turning
regcv.fit(X, y)

# if you want to know best parametes and best score
print(regcv.best_score_)
print(regcv.best_params_)
