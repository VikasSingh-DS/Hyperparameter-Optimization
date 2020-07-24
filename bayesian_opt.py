import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization


from sklearn import model_selection
import xgboost


df = pd.read_csv("datasets_train.csv")
X = df.drop("price_range", axis=1).values  # input feature
y = df.price_range.values  # target feature

dtrain = xgboost.DMatrix(data=X, label=y)


def xgb_evaluate(max_depth, gamma, colsample_bytree, eta, n_estimators):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": int(max_depth),
        "subsample": 0.8,
        "n_estimators": int(n_estimators),
        "eta": eta,
        "gamma": gamma,
        "colsample_bytree": colsample_bytree,
    }
    # Used around 100 boosting rounds in the full model
    cv_result = xgboost.cv(
        params, dtrain, num_boost_round=100, nfold=3, stratified=True
    )

    # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE
    return cv_result["test-auc-mean"].max()


xgb_bo = BayesianOptimization(
    xgb_evaluate,
    {
        "max_depth": (3, 10),
        "gamma": (0, 1),
        "eta": (0.01, 0.2),
        "n_estimators": (1000, 2000),
        "colsample_bytree": (0.4, 0.95),
    },
)
xgb_bo.maximize(init_points=5, n_iter=25)

print(xgb_bo.max)

