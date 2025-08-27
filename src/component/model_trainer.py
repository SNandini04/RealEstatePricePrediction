
import os
import argparse
import json
from datetime import datetime
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib

# Try to import xgboost if available
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


def load_data(path: str) -> pd.DataFrame:
    """Load CSV data into a pandas DataFrame."""
    df = pd.read_csv(path)
    print(f"[INFO] Loaded data shape: {df.shape}")
    return df


def prepare_xy(
    df: pd.DataFrame,
    target: str,
    drop_cols: list = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Basic preparation to split features and target.
    By default it drops columns that are non-feature like ids or raw text columns if provided.
    """
    df = df.copy()
    drop_cols = drop_cols or []
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe columns.")
    X = df.drop(columns=[target] + drop_cols, errors='ignore')
    y = df[target].copy()
    return X, y


def make_preprocessing_pipeline(
    X: pd.DataFrame,
    numeric_impute_strategy: str = "median",
    categorical_impute_strategy: str = "most_frequent",
    numeric_scaler: Any = StandardScaler(),
    pca_components: int = None,
) -> Tuple[ColumnTransformer, list]:
    """
    Build a ColumnTransformer pipeline for numeric and categorical processing.
    Returns the ColumnTransformer and output feature names (approximate).
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=numeric_impute_strategy)),
        ('scaler', numeric_scaler)
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=categorical_impute_strategy)),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocess = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ], remainder='drop')

    # feature names: for reporting, a best-effort list
    feature_names = numeric_cols.copy()
    # append placeholder names for OHE columns (can't get exact names until fitted)
    if categorical_cols:
        feature_names += [f"ohe_{c}" for c in categorical_cols]

    return preprocess, feature_names


def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """Return a dictionary of candidate models (unfitted)."""
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(random_state=random_state),
        'Lasso': Lasso(random_state=random_state),
        'DecisionTree': DecisionTreeRegressor(random_state=random_state),
        'RandomForest': RandomForestRegressor(random_state=random_state, n_jobs=-1),
        'KNeighbors': KNeighborsRegressor(),
        'SVR': SVR(),
    }
    if _HAS_XGB:
        models['XGBoost'] = XGBRegressor(random_state=random_state, objective='reg:squarederror', n_jobs=-1)
    return models


def evaluate_model_cv(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: list = None,
) -> Dict[str, float]:
    """
    Cross-validate the pipeline and return mean metrics: MAE, RMSE, R2.
    scoring parameter supporting ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']
    """
    if scoring is None:
        scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']

    cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False, n_jobs=-1)
    mae = -np.mean(cv_results['test_neg_mean_absolute_error'])
    mse = -np.mean(cv_results['test_neg_mean_squared_error'])
    rmse = np.sqrt(mse)
    r2 = np.mean(cv_results['test_r2'])
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}


def run_model_comparisons(
    X: pd.DataFrame,
    y: pd.Series,
    preprocess: ColumnTransformer,
    models: Dict[str, Any],
    cv: int = 5,
) -> pd.DataFrame:
    """Run CV for each candidate model and return a DataFrame of results."""
    results = []
    for name, model in models.items():
        print(f"[INFO] Evaluating: {name}")
        pipe = Pipeline(steps=[('preprocess', preprocess), ('model', model)])
        metrics = evaluate_model_cv(pipe, X, y, cv=cv)
        results.append({
            'model': name,
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'R2': metrics['R2']
        })
        print(f"  -> MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, R2: {metrics['R2']:.4f}")
    return pd.DataFrame(results).sort_values(by='R2', ascending=False).reset_index(drop=True)


def tune_model_gridsearch(
    X: pd.DataFrame,
    y: pd.Series,
    preprocess: ColumnTransformer,
    model,
    param_grid: dict,
    cv: int = 5,
    scoring: str = 'neg_mean_absolute_error',
    n_jobs: int = -1,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Run GridSearchCV over pipeline(preprocess + model). Returns best_estimator_ and best_params_.
    """
    pipe = Pipeline(steps=[('preprocess', preprocess), ('model', model)])
    grid = GridSearchCV(pipe, param_grid={'model__' + k: v for k, v in param_grid.items()},
                        cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=1)
    print("[INFO] Starting GridSearchCV ...")
    grid.fit(X, y)
    print(f"[INFO] Best score (grid search): {grid.best_score_}, Best params: {grid.best_params_}")
    return grid.best_estimator_, grid.best_params_


def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    best_pipeline: Pipeline,
) -> Dict[str, Any]:
    """Fit pipeline (preprocess + model) on training data and evaluate on validation set."""
    print("[INFO] Training final model on training data ...")
    best_pipeline.fit(X_train, y_train)
    print("[INFO] Predicting on validation set ...")
    preds = best_pipeline.predict(X_valid)
    mae = mean_absolute_error(y_valid, preds)
    rmse = mean_squared_error(y_valid, preds, squared=False)
    r2 = r2_score(y_valid, preds)
    print(f"[RESULT] Validation MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'preds': preds}


def save_model(pipeline: Pipeline, path: str):
    """Save the trained pipeline/model to disk using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"[INFO] Saved model pipeline to: {path}")


def save_results_df(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[INFO] Saved results to: {path}")


def main(args):
    # 1. Load data
    df = load_data(args.data)

    # 2. Prep X, y
    drop_cols = args.drop_cols.split(',') if args.drop_cols else []
    X, y = prepare_xy(df, target=args.target, drop_cols=drop_cols)

    # 3. Create preprocess pipeline
    preprocess, feature_names = make_preprocessing_pipeline(X)

    # 4. Candidate models
    models = get_candidate_models()
    print(f"[INFO] Candidate models: {list(models.keys())}")

    # 5. Compare models with CV
    print("[INFO] Running cross-validation comparisons ...")
    results_df = run_model_comparisons(X, y, preprocess, models, cv=args.cv)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or "outputs"
    result_file = os.path.join(output_dir, f"model_comparison_{timestamp}.csv")
    save_results_df(results_df, result_file)

    # 6. If tuning requested, tune the chosen model
    if args.tune and args.tune_model:
        model_name = args.tune_model
        if model_name not in models:
            raise ValueError(f"Model {model_name} not in candidate list.")
        model_to_tune = models[model_name]

        # Example param grids - you may adjust these to match your notebook tunings
        param_grids = {
            'RandomForest': {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]},
            'XGBoost': {'n_estimators': [100, 200], 'max_depth': [6, 10], 'learning_rate': [0.05, 0.1]},
            'SVR': {'C': [1, 10], 'gamma': ['scale', 'auto']},
            'KNeighbors': {'n_neighbors': [3, 5, 7]},
            'Ridge': {'alpha': [0.1, 1, 10]},
            'Lasso': {'alpha': [0.001, 0.01, 0.1]},
            'DecisionTree': {'max_depth': [6, 12, None]}
        }
        grid = param_grids.get(model_name, {'n_estimators': [100, 200]})
        best_pipe, best_params = tune_model_gridsearch(X, y, preprocess, model_to_tune, grid, cv=args.cv)
        # Save best pipeline
        save_model(best_pipe, os.path.join(output_dir, f"best_pipeline_{model_name}_{timestamp}.pkl"))
    else:
        best_pipe = None
        best_params = None

    # 7. Train final model (either tuned or top result from CV)
    # Split train/valid for final check
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.valid_size, random_state=42)
    if best_pipe is None:
        # pick top model from results_df
        top_model_name = results_df.loc[0, 'model']
        print(f"[INFO] No tuning requested - training top model from CV: {top_model_name}")
        model_obj = models[top_model_name]
        best_pipe = Pipeline(steps=[('preprocess', preprocess), ('model', model_obj)])
        best_pipe.fit(X_train, y_train)

    final_metrics = train_final_model(X_train, y_train, X_valid, y_valid, best_pipe)

    # 8. Save final pipeline and metrics
    os.makedirs(output_dir, exist_ok=True)
    final_model_path = os.path.join(output_dir, f"final_model_{timestamp}.pkl")
    save_model(best_pipe, final_model_path)

    metrics_out = {
        'timestamp': timestamp,
        'model': args.tune_model if best_params else results_df.loc[0, 'model'],
        'best_params': json.dumps(best_params) if best_params else None,
        'MAE': float(final_metrics['MAE']),
        'RMSE': float(final_metrics['RMSE']),
        'R2': float(final_metrics['R2'])
    }
    metrics_df = pd.DataFrame([metrics_out])
    save_results_df(metrics_df, os.path.join(output_dir, f"final_metrics_{timestamp}.csv"))

    print("[DONE]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model selection and tuning for RealEstatePricePrediction")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to processed CSV file (e.g. gurgaon_properties_post_feature_selection_v2.csv)")
    parser.add_argument("--target", type=str, default="price", help="Target column name")
    parser.add_argument("--drop_cols", type=str, default="", help="Comma-separated columns to drop (optional)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save models and results")
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--tune", action='store_true', help="Whether to run GridSearchCV tuning for a model")
    parser.add_argument("--tune_model", type=str, default=None, help="Name of the model to tune (e.g. RandomForest)")
    parser.add_argument("--valid_size", type=float, default=0.2, help="Validation split size for final evaluation")
    args = parser.parse_args()
    main(args)
