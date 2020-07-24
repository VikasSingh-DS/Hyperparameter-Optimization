import pandas as pd
import numpy as np

from sklearn import model_selection, metrics
import xgboost
from functools import partial
import optuna


df = pd.read_csv("datasets_train.csv")
X = df.drop("price_range", axis=1).values  # input feature
y = df.price_range.values  # target feature


def optimize(trial, x, y):
    max_depth = trial.suggest_int("max_depth", 5, 25)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
    eta = trial.suggest_uniform("eta", 0.01, 0.2)
    gamma = trial.suggest_int("gamma", 0.01, 0.2)

    xgb_opt = xgboost.XGBClassifier(
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        eta=eta,
        gamma=gamma,
        tree_method="gpu_hist",
    )
    kfold = model_selection.StratifiedKFold(n_splits=5, shuffle=True)
    acc = []
    for train_ix, test_ix in kfold.split(X, y):
        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]

        xgb_opt.fit(train_X, train_y)
        preds = xgb_opt.predict(test_X)
        fold_acc = metrics.accuracy_score(test_y, preds)
        acc.append(fold_acc)

    return -1.0 * np.mean(acc)


optimization_function = partial(optimize, x=X, y=y)

study = optuna.create_study(direction="minimize")
study.optimize(optimization_function, n_trails=15)
study.best_params
