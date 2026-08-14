# -*- coding: utf-8 -*-

import os
import sys

OUTPUT_DIR = "results_2.4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() 

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

sys.stdout = Logger(f"{OUTPUT_DIR}/run.log")

# =============================================================================
# 0. Imports
# =============================================================================
import numpy as np
import pandas as pd
import pyreadstat
import statistics
import statsmodels.api as sm
import statsmodels.stats.api as sms
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import scikit_posthocs as sp
from scipy.stats import shapiro, levene
from scipy import stats
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error, r2_score

# =============================================================================
# 1. Load data
# =============================================================================
df, meta = pyreadstat.read_sav("spss_merged_grade4_student.sav")
df.shape
df.head()

# =============================================================================
# 2. Define variables
# =============================================================================

# Linguistic variables
linguistic_vars = [
    "ASBHELA", "ASDHELA",
    "ASBHENA", "ASDHENA",
    "ASBHELN", "ASDHELN",
    "ASBHELT", "ASDHELT",
    "ASBHENT", "ASDHENT",
    "ASBHLNT", "ASDHLNT",
    "ASBHPSP", "ASDHPSP",
]

# Capital variables
capital_vars = [
    "ASBH01A","ASBH01B","ASBH01C","ASBH01D","ASBH01E","ASBH01F","ASBH01G",
    "ASBH01H","ASBH01I","ASBH01J","ASBH01K","ASBH01L","ASBH01M","ASBH01N",
    "ASBH01O","ASBH01P","ASBH01Q","ASBH01R",
    "ASBH02A",
    "ASBH03A","ASBH03B","ASBH03C","ASBH03D","ASBH03E","ASBH03F",
    "ASBH04AA","ASBH04AB","ASBH04B",
    "ASBH05",
    "ASBH06A","ASBH06B","ASBH06C","ASBH06D","ASBH06E","ASBH06F","ASBH06G",
    "ASBH07A","ASBH07B","ASBH07C","ASBH07D","ASBH07E",
    "ASBH08A","ASBH08B","ASBH08C","ASBH08D","ASBH08E","ASBH08F","ASBH08G","ASBH08H",
    "ASBH09A","ASBH09B","ASBH09C","ASBH09D","ASBH09E",
    "ASBH10","ASBH11",
    "ASBH12A","ASBH12B","ASBH12C",
    "ASBH13A","ASBH13B",
    "ASBH14AA","ASBH14AB","ASBH14BA","ASBH14BB",
    "ASBH14CA","ASBH14CB",
    "ASBH14DA","ASBH14DB",
    "ASBH14EA","ASBH14EB",
    "ASBH14FA","ASBH14FB",
    "ASBH15",
    "ASBH16A","ASBH16B",
    "ASBH17",
    "ASBH18A","ASBH18B",
    "ASBHSES","ASDHSES",
    "ASDHOCCP",
    "ASDHEDUP",
    "ASDHAPS",
]
features_numeric = ["ASBHELA", "ASBHELT", "ASBHENA", "ASBHENT", "ASBHLNT", "ASBHPSP", "ASBHSES", "ASBHELN"]
features_categorical = ["ASBH01A", "ASBH01B","ASBH01C","ASBH01D","ASBH01E","ASBH01F","ASBH01G",
    "ASBH01H","ASBH01I","ASBH01J","ASBH01K","ASBH01L","ASBH01M","ASBH01N",
    "ASBH01O","ASBH01P","ASBH01Q","ASBH01R",
    "ASBH02A",
    "ASBH03A","ASBH03B","ASBH03C","ASBH03D","ASBH03E","ASBH03F",
    "ASBH04AA","ASBH04AB","ASBH04B",
    "ASBH05",
    "ASBH06A","ASBH06B","ASBH06C","ASBH06D","ASBH06E","ASBH06F","ASBH06G",
    "ASBH07A","ASBH07B","ASBH07C","ASBH07D","ASBH07E",
    "ASBH08A","ASBH08B","ASBH08C","ASBH08D","ASBH08E","ASBH08F","ASBH08G","ASBH08H",
    "ASBH09A","ASBH09B","ASBH09C","ASBH09D","ASBH09E",
    "ASBH10","ASBH11",
    "ASBH12A","ASBH12B","ASBH12C",
    "ASBH13A","ASBH13B",
    "ASBH14AA","ASBH14AB","ASBH14BA","ASBH14BB",
    "ASBH14CA","ASBH14CB",
    "ASBH14DA","ASBH14DB",
    "ASBH14EA","ASBH14EB",
    "ASBH14FA","ASBH14FB",
    "ASBH15",
    "ASBH16A","ASBH16B",
    "ASBH17",
    "ASBH18A","ASBH18B", "ASDHELA", "ASDHENA", "ASDHELN", "ASDHELT", "ASDHENT", "ASDHLNT", "ASDHPSP", "ASDHSES", "ASDHOCCP", "ASDHEDUP", "ASDHAPS" ]

# Target + Features
target = ["ASMMAT01","ASMMAT02","ASMMAT03","ASMMAT04","ASMMAT05"]
weight = "TOTWGT"
features = linguistic_vars + capital_vars

X = df[features]
y = df[target]
w = df[weight]
X.shape, y.shape
X_numeric = df[features_numeric]
X_categorical = df[features_categorical]
print(X.shape)

# data imputation (preprocess only in features) KNN
print(X.shape)
X_KNN_imputed = X.copy(deep=True)
# Create KNN imputer
knn_imputer = KNNImputer(n_neighbors=5)
print(X_KNN_imputed[features_numeric].isna().sum())
X_KNN_imputed[features_numeric] = knn_imputer.fit_transform(X_KNN_imputed[features_numeric])
print(X_KNN_imputed[features_numeric].isna().sum())
print(X.shape)
print(X_KNN_imputed.shape)

print(X.shape)
print(X_KNN_imputed[features_categorical].isna().sum())
X_KNN_imputed[features_categorical] = X_KNN_imputed[features_categorical].fillna(X_KNN_imputed[features_categorical].mode().iloc[0])
print(X_KNN_imputed[features_categorical].isna().sum())
print(X.shape)
print(X_KNN_imputed.shape)

df_imputed = df.copy()
df_imputed[features] = X_KNN_imputed[features]

pyreadstat.write_sav(df_imputed, "spss_merged_grade4_student_imputed.sav")

# one-hot encoding (preprocess only in features)
# apply dummy only to categorical ones
X_KNN_imputed = pd.get_dummies(X_KNN_imputed, columns=features_categorical, drop_first=True)
X_KNN_imputed.head()
print(X_KNN_imputed.shape)

# =============================================================================
# Linearity Check
# =============================================================================

for t in target:
    print(f"\n==============================")
    print(f"OLS Linearity Diagnostics for {t}")
    print(f"==============================")

    y_t = y[t]

    # Convert boolean columns to integer (0 or 1) to avoid statsmodels error
    X_KNN_imputed_numeric = X_KNN_imputed.copy()
    for col in X_KNN_imputed_numeric.select_dtypes(include='bool').columns:
        X_KNN_imputed_numeric[col] = X_KNN_imputed_numeric[col].astype(int)

    Xc = sm.add_constant(X_KNN_imputed_numeric)

    model = sm.OLS(y_t, Xc).fit()

    # 1) Residuals vs Fitted
    fitted = model.fittedvalues
    resid = model.resid

    plt.figure(figsize=(6,4))
    plt.scatter(fitted, resid, alpha=0.4)
    plt.axhline(0, color='red', lw=1)
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title(f"Residuals vs Fitted ({t})")

    plt.savefig(
        f"{OUTPUT_DIR}/Residuals_{t}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # 2) Ramsey RESET test
    reset = sms.linear_reset(model, power=2, use_f=True)
    print("Ramsey RESET test:")
    print(reset)

    # 3) Partial residual plot for one numeric feature
    feature = features_numeric[7]
    beta_j = model.params[feature]
    partial_resid = resid + beta_j * X_KNN_imputed_numeric[feature]
    plt.figure(figsize=(6,4))
    plt.scatter(X_KNN_imputed_numeric[feature], partial_resid, alpha=0.4)
    plt.xlabel(feature)
    plt.ylabel("Partial residual")
    plt.title(f"Partial residual plot: {feature} ({t})")

    plt.savefig(
        f"{OUTPUT_DIR}/PartialResiduals_{t}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

def run_model_pipeline(model_name, model, param_grid,
                        X_train, X_test, y_train, y_test, w_train):

    scoring = {
        'mse': 'neg_mean_squared_error',
        'rmse': 'neg_root_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
        'r2': 'r2'
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        refit='rmse',
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    if model_name == "KNN":
        grid.fit(X_train, y_train)
    else:
        grid.fit(X_train, y_train, sample_weight=w_train)

    best_model = grid.best_estimator_
    best_idx = grid.best_index_

    # Train 
    train_pred = best_model.predict(X_train)
    train_mse = MSE(y_train, train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)

    # Validation 
    cv_mse  = -grid.cv_results_['mean_test_mse'][best_idx]
    cv_rmse = -grid.cv_results_['mean_test_rmse'][best_idx]
    cv_mae  = -grid.cv_results_['mean_test_mae'][best_idx]
    cv_r2   =  grid.cv_results_['mean_test_r2'][best_idx]

    # Test 
    test_pred = best_model.predict(X_test)
    test_mse = MSE(y_test, test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)

    # Feature importance
    importances = None
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
    elif hasattr(best_model, "coef_"):
        importances = pd.Series(best_model.coef_, index=X_train.columns)
        
    return {
        "results": {
            "Train": (train_mse, train_rmse, train_mae, train_r2),
            "Validation": (cv_mse, cv_rmse, cv_mae, cv_r2),
            "Test": (test_mse, test_rmse, test_mae, test_r2)
        },
        "best_params": grid.best_params_,
        "feature_importances": importances
    }

rf_grid = {
    'n_estimators': [200, 350, 400],
    'max_features': ['sqrt', 'log2', 0.5, 0.8],
    'max_depth': [4, 6, 8],
    'min_samples_leaf': [5, 10, 20],
}

xgb_grid = {
    'n_estimators': [200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [2, 3, 4],
    'subsample': [0.6, 0.7, 0.8],
    'colsample_bytree': [0.3, 0.5, 0.7],
}

lgbm_grid = {
    'n_estimators': [200, 300, 500],
    'learning_rate': [0.03, 0.05, 0.1],
    'num_leaves': [7, 15, 31],
    'min_data_in_leaf': [10, 20, 50],
    'subsample': [0.6, 0.7, 0.8],
}

targets = ["ASMMAT01", "ASMMAT02", "ASMMAT03", "ASMMAT04", "ASMMAT05"]
all_results = []
all_best_params = {}
all_feature_importances = {}

seeds_list = [1, 42, 123, 888, 2026]

for SEED in seeds_list:
    print(f"\n###########################################")
    print(f"Start Running Random SEED = {SEED} ")
    print(f"###########################################")

    all_best_params[SEED] = {}
    all_feature_importances[SEED] = {}

    for target in targets:
        print("\n==============================")
        print("Running PV:", target)
        print("==============================")
        y = df[target]
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X_KNN_imputed, y, w, test_size=0.2, random_state=SEED
        )
        # Standardization
        scaler = StandardScaler()
        numeric_cols = features_numeric
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

        all_best_params[SEED][target] = {}
        all_feature_importances[SEED][target] = {}

        # RF
        rf_out = run_model_pipeline(
            "RandomForest",
            RandomForestRegressor(random_state=SEED),
            rf_grid,
            X_train_scaled, X_test_scaled, y_train, y_test, w_train
        )
        for split, (mse, rmse, mae, r2) in rf_out["results"].items():
            all_results.append([SEED, target, "RandomForest", split, mse, rmse, mae, r2])
        all_best_params[SEED][target]["RandomForest"] = rf_out["best_params"]
        all_feature_importances[SEED][target]["RandomForest"] = rf_out["feature_importances"]

        # XGB
        xgb_out = run_model_pipeline(
            "XGBoost",
            XGBRegressor(objective='reg:squarederror', random_state=SEED, eval_metric="rmse"),
            xgb_grid,
            X_train_scaled, X_test_scaled, y_train, y_test, w_train
        )
        for split, (mse, rmse, mae, r2) in xgb_out["results"].items():
            all_results.append([SEED, target, "XGBoost", split, mse, rmse, mae, r2])
        all_best_params[SEED][target]["XGBoost"] = xgb_out["best_params"]
        all_feature_importances[SEED][target]["XGBoost"] = xgb_out["feature_importances"]

        # LGBM
        lgbm_out = run_model_pipeline(
            "LightGBM",
            LGBMRegressor(objective='regression', random_state=SEED),
            lgbm_grid,
            X_train_scaled, X_test_scaled, y_train, y_test, w_train
        )
        for split, (mse, rmse, mae, r2) in lgbm_out["results"].items():
            all_results.append([SEED, target, "LightGBM", split, mse, rmse, mae, r2])
        all_best_params[SEED][target]["LightGBM"] = lgbm_out["best_params"]
        all_feature_importances[SEED][target]["LightGBM"] = lgbm_out["feature_importances"]

print("NaN in X_train_scaled:", X_train_scaled.isna().sum().sum())
print("NaN in X_test_scaled:", X_test_scaled.isna().sum().sum())

print("Before scaling NaN:", X_train.isna().sum().sum())
print("After scaling NaN:", X_train_scaled.isna().sum().sum())
print(X_train_scaled.isna().sum().sort_values(ascending=False).head(20))

results_df = pd.DataFrame(
    all_results,
    columns=["Seed", "PV", "Model", "Split", "MSE", "RMSE", "MAE", "R2"]
)

with open(f"{OUTPUT_DIR}/best_params.json", "w") as f:
    json.dump(all_best_params, f, indent=4)

print("\n=== Comprehensive Results Across PVs and Models ===")
print(results_df.round(4).to_string(index=False))

print("\n=== Best Parameters per SEED and PV ===")
for seed in all_best_params:
    print(f"\n################################### Seed: {seed} ###################################")
    for pv in all_best_params[seed]:
        print(f"\n  PV: {pv}")
        for model in all_best_params[seed][pv]:
            print(f"    {model} : {all_best_params[seed][pv][model]}")

print("\n=== Feature Importances per SEED and PV (Top 10) ===")
for seed in all_feature_importances:
    print(f"\n=================================== Seed: {seed} ===================================")
    for pv in all_feature_importances[seed]:
        print(f"\n  PV: {pv}")
        for model in all_feature_importances[seed][pv]:
            print(f"\n    Model: {model}")
            importance_series = all_feature_importances[seed][pv][model]
            if isinstance(importance_series, pd.Series):
                print(importance_series.sort_values(ascending=False).head(10))
            else:
                print(importance_series)

for seed in all_feature_importances:
    for pv in all_feature_importances[seed]:
        for model in all_feature_importances[seed][pv]:
            importance_series = all_feature_importances[seed][pv][model]
            if isinstance(importance_series, pd.Series):
                fi = importance_series.sort_values(ascending=False).head(10)

                plt.figure(figsize=(8,6))
                fi.plot(kind="barh")
                plt.title(f"Top 10 Feature Importance\nSeed={seed} | PV={pv} | Model={model}")
                plt.gca().invert_yaxis()
                plt.xlabel("Importance")
                plt.tight_layout()

                plt.savefig(
                    f"{OUTPUT_DIR}/FI_{seed}_{pv}_{model}.png",
                    dpi=300,
                    bbox_inches="tight"
                )
                plt.close()

results_df.to_csv(
    f"{OUTPUT_DIR}/results.csv",
    index=False
)

summary_mean_std = (
    results_df
    .groupby(["PV", "Model", "Split"], as_index=False)
    .agg(
        MSE_mean=("MSE", "mean"),
        MSE_std=("MSE", "std"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        R2_mean=("R2", "mean"),
        R2_std=("R2", "std")
    )
)

summary_mean_std.to_csv(
    f"{OUTPUT_DIR}/summary_mean_std.csv",
    index=False
)

# Set custom ordering for Split
split_order = pd.CategoricalDtype(["Train", "Validation", "Test"], ordered=True)
summary_mean_std["Split"] = summary_mean_std["Split"].astype(split_order)
summary_mean_std = summary_mean_std.sort_values(["PV", "Model", "Split"]).reset_index(drop=True)

print("\n=== Summary Mean & STD by PV, Model, Split ===")
print(summary_mean_std)

# Compute standard deviation for training and test targets
std_train = statistics.stdev(y_train)
std_test = statistics.stdev(y_test)

print("\nStandard deviation of y_train:", round(std_train, 4))
print("Standard deviation of y_test:", round(std_test, 4))

# Dynamically extract mean Test RMSE for each model
rmse_rf_test = summary_mean_std[(summary_mean_std["Model"] == "RandomForest") & (summary_mean_std["Split"] == "Test")]["RMSE_mean"].mean()
rmse_xgbm_test = summary_mean_std[(summary_mean_std["Model"] == "XGBoost") & (summary_mean_std["Split"] == "Test")]["RMSE_mean"].mean()
rmse_lgbm_test = summary_mean_std[(summary_mean_std["Model"] == "LightGBM") & (summary_mean_std["Split"] == "Test")]["RMSE_mean"].mean()

print("\nRF Test RMSE / SD(y_test):", rmse_rf_test / std_test)
print("XGBoost Test RMSE / SD(y_test):", rmse_xgbm_test / std_test)
print("LightGBM Test RMSE / SD(y_test):", rmse_lgbm_test / std_test)

# =============================================================================
# Normality & Variance Tests (Shapiro–Wilk & Levene)
# =============================================================================

data = {}
for model_name in results_df["Model"].unique():
    data[model_name] = {
        "train": results_df[(results_df["Model"] == model_name) & (results_df["Split"] == "Train")]["RMSE"].tolist(),
        "val":   results_df[(results_df["Model"] == model_name) & (results_df["Split"] == "Validation")]["RMSE"].tolist(),
        "test":  results_df[(results_df["Model"] == model_name) & (results_df["Split"] == "Test")]["RMSE"].tolist()
    }

data_df = pd.DataFrame({
    (model, split): values
    for model, splits in data.items()
    for split, values in splits.items()
})

data_df.to_csv(f"{OUTPUT_DIR}/data.csv", index=False)

def check_normality(metric):
    print(f"\n=== Shapiro–Wilk Normality Test for {metric.upper()} ===")
    for model, metrics in data.items():
        stat, p = shapiro(metrics[metric])
        print(f"{model:12s}  p = {p:.4f}")

def check_variance(metric):
    print(f"\n=== Levene’s Test for {metric.upper()} (Homogeneity of Variance) ===")
    groups = [metrics[metric] for metrics in data.values()]
    stat, p = levene(*groups)
    print(f"Levene p = {p:.4f}")

for metric in ["train", "val", "test"]:
    check_normality(metric)
    check_variance(metric)

# =============================================================================
# Pivot Table & Friedman Test (Paired Non-parametric Test & Conover Post-hoc)
# =============================================================================

test_results = results_df[results_df["Split"] == "Test"].copy()

test_results.to_csv(
    f"{OUTPUT_DIR}/test_results.csv",
    index=False
)

pivot_df = test_results.pivot_table(index=["Seed", "PV"], columns="Model", values="RMSE")

print("\nReshaped Data Matrix (Sample):")
print(pivot_df.head())

pivot_df.to_csv(
    f"{OUTPUT_DIR}/pivot_df.csv",
    index=False
)

rf_scores = pivot_df["RandomForest"].dropna().values
xgb_scores = pivot_df["XGBoost"].dropna().values
lgbm_scores = pivot_df["LightGBM"].dropna().values

stat, p_value = stats.friedmanchisquare(rf_scores, xgb_scores, lgbm_scores)

print("\n==============================")
print("Friedman Test Result")
print("==============================")
print(f"Statistic: {stat:.4f}")
print(f"p-value: {p_value:.4e}")

# =============================================================================
# Post-hoc Test 
# =============================================================================
if p_value < 0.05:
    print("There are significant differences in the predictive performance of these models.")
    print("\n=== Conover Post-hoc Test ===")
    posthoc_conover = sp.posthoc_conover_friedman(
        pivot_df.values, 
        p_adjust='holm'
    )

    posthoc_conover.columns = pivot_df.columns
    posthoc_conover.index = pivot_df.columns
    print(posthoc_conover.round(4))

    posthoc_conover.to_csv(
        f"{OUTPUT_DIR}/posthoc_conover.csv",
        index=True
    )

else:
    print("There are no significant differences in the predictive performance of these models.")

sys.stdout.close()
