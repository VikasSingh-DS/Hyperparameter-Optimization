import pandas as pd
import numpy as np

from sklearn import model_selection, metrics
import xgboost
from functools import partial
from skopt import space, gp_minimize


df = pd.read_csv("datasets_train.csv")
X = df.drop("price_range", axis=1).values  # input feature
y = df.price_range.values  # target feature


def optimize(params, param_names, x, y):
    params = dict(zip(param_names, params))
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


param_space = [
    space.Integer(3, 15, name="max_depth"),
    space.Integer(1, 10, name="min_child_weight"),
    space.Real(0.1, 0.3, prior="uniform", name="eta"),
    space.Real(0.5, 0.7, prior="uniform", name="subsample"),
    space.Real(0.1, 0.5, prior="uniform", name="gamma"),
]

param_names = ["max_depth", "min_child_weight", "eta", "subsample", "gamma"]
optimization_function = partial(optimize, param_names=param_names, x=X, y=y)

result = gp_minimize(
    optimization_function,
    dimensions=param_space,
    n_calls=15,
    n_random_starts=10,
    verbose=10,
)

print(dict(zip(param_names, result.x)))

