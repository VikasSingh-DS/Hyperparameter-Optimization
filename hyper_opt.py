import pandas as pd
import numpy as np

from sklearn import model_selection, metrics
import xgboost
from functools import partial
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope

df = pd.read_csv("datasets_train.csv")
X = df.drop("price_range", axis=1).values  # input feature
y = df.price_range.values  # target feature


def optimize(params, x, y):
    xgb_opt = xgboost.XGBClassifier(**params, tree_method="gpu_hist")
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


space = {
    "max_depth": scope.int(hp.quniform("max_depth", 5, 25, 1)),
    "min_child_weight": scope.int(hp.quniform("min_child_weight", 1, 10, 1)),
    "eta": hp.uniform("eta", 0.01, 0.2),
    "subsample": hp.choice("subsample", [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    "gamma": hp.uniform("gamma", 0.01, 0.2),
}

params = ["max_depth", "min_child_weight", "eta", "subsample", "gamma"]


optimization_function = partial(optimize, x=X, y=y)

trials = Trials()

result = fmin(
    fn=optimization_function,
    space=space,
    max_evals=15,
    trials=trials,
    algo=tpe.suggest,
)

print(result)
