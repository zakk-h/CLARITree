import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

from clari_tree import CLARITree


def report(name, y_true, pred):
    print(f"\n{name}")
    print("MAE: ", mean_absolute_error(y_true, pred))
    print("RMSE:", mean_squared_error(y_true, pred) ** 0.5)
    print("R^2: ", r2_score(y_true, pred))


data = fetch_california_housing()
X = data.data.astype(float)
y = data.target.astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

ref_model = XGBRegressor(
    n_estimators=2000,
    max_depth=4,
    learning_rate=0.015,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=3,
    reg_lambda=5.0,
    reg_alpha=0.0,
    objective="reg:squarederror",
    random_state=0,
    n_jobs=-1,
)

ref_model.fit(X_train, y_train)

ref_train = ref_model.predict(X_train)
ref_test = ref_model.predict(X_test)

report("Reference XGBoost train", y_train, ref_train)
report("Reference XGBoost test", y_test, ref_test)

sample_weight = np.ones_like(y_train, dtype=float)
sample_weight[y_train >= np.quantile(y_train, 0.75)] = 5.0

print("\nSample weight summary")
print("min:", sample_weight.min())
print("max:", sample_weight.max())
print("mean:", sample_weight.mean())
print("num high-weight:", np.sum(sample_weight > 1.0))

tree_unweighted = CLARITree(
    kappa=0.01,
    depth=5,
    lambda_=0.005,
    rho=0.05,
    eta=0.50,
    n_thresholds=30,
    thresholds_strategy="quantile",
    verbose=True,
    min_leaf_node_size=50,
)

obj_unweighted = tree_unweighted.fit_with_reference(X_train, y_train, ref_train)
pred_unweighted = tree_unweighted.predict_with_reference(X_test, ref_test)

report("Unweighted three-leaf CLARITree test", y_test, pred_unweighted)

print("\nUnweighted objective:", obj_unweighted)
print(tree_unweighted.print_tree())

tree_weighted = CLARITree(
    kappa=0.01,
    depth=5,
    lambda_=0.005,
    rho=0.05,
    eta=0.50,
    n_thresholds=30,
    thresholds_strategy="quantile",
    verbose=True,
    min_leaf_node_size=50,
)

obj_weighted = tree_weighted.fit_with_reference_and_weights(
    X_train,
    y_train,
    ref_train,
    sample_weight,
)

pred_weighted = tree_weighted.predict_with_reference(X_test, ref_test)

report("Weighted three-leaf CLARITree test", y_test, pred_weighted)

high_test = y_test >= np.quantile(y_train, 0.75)
report("Unweighted CLARITree high-y test subset", y_test[high_test], pred_unweighted[high_test])
report("Weighted CLARITree high-y test subset", y_test[high_test], pred_weighted[high_test])

print("\nWeighted objective:", obj_weighted)
print(tree_weighted.print_tree())