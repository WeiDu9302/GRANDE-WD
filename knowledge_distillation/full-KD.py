import numpy as np
import pandas as pd
import xgboost as xg
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
import warnings
import os  
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import sys
import json
import itertools

_HAS_GRADTREE = False
_HAS_GRANDE = False

try:
    from GradTree.GradTree import GradTree
    _HAS_GRADTREE = True
except ImportError:
    print("Warning: GradTree library not detected, GradTree student model will be skipped.")
    _HAS_GRADTREE = False

try:
    from GRANDE import GRANDE
    _HAS_GRANDE = True
except ImportError:
    print("Warning: GRANDE library not detected, GRANDE student model will be skipped.")
    _HAS_GRANDE = False

from pytorch_tabnet.tab_model import TabNetRegressor

# -----------------------------------------
# Suppress warnings for cleaner output
# -----------------------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats._stats_py")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")
warnings.filterwarnings("ignore", message="X does not have valid feature names, but OneHotEncoder was fitted with feature names")
warnings.filterwarnings("ignore", message="Device used : cpu")
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_tabnet.callbacks") # Suppress TabNet early stopping warning

# --- Helper class for redirecting stdout/stderr to file and console ---
class OutputDuplicator:
    def __init__(self, original_stream, log_file_handle):
        self.original_stream = original_stream
        self.log_file_handle = log_file_handle

    def write(self, message):
        self.original_stream.write(message)
        if self.log_file_handle and not self.log_file_handle.closed:
            try:
                self.log_file_handle.write(message)
            except Exception as e:
                self.original_stream.write(f"\nError writing to log file: {e}\n")

    def flush(self):
        self.original_stream.flush()
        if self.log_file_handle and not self.log_file_handle.closed:
            try:
                self.log_file_handle.flush()
            except Exception:
                pass

    def __getattr__(self, attr):
        return getattr(self.original_stream, attr)

# --- Helper function to patch instance with __sklearn_tags__ ---
def patch_instance_sklearn_tags(obj):
    try:
        setattr(obj, "__sklearn_tags__", None)
    except Exception:
        pass

# -------------------------
# List of all datasets to process
# -------------------------
DATASETS = [
    {"name": "pol", "id": 201},
    {"name": "elevators", "id": 216},
    {"name": "isolet", "id": 300},
    {"name": "wine_quality", "id": 287},
    {"name": "Ailerons", "id": 296},
    {"name": "houses", "id": 537},
    {"name": "house_16H", "id": 574},
    {"name": "diamonds", "id": 42225},
    {"name": "Brazilian_houses", "id": 42688},
    {"name": "Bike_Sharing_Demand", "id": 42712},
    {"name": "nyc_taxi_green_dec_2016", "id": 42729},
    {"name": "house_sales", "id": 42731},
    {"name": "sulfur", "id": 23515},
    {"name": "medical_charges", "id": 42720},
    {"name": "MiamiHousing2016", "id": 43093},
    {"name": "superconduct", "id": 43174},
    {"name": "cpu_act", "id": 197},
]

def calculate_metrics(y_true, y_pred, model_name="Model"):
    """Calculate and print regression metrics, automatically handle possible NaN in predictions."""
    if y_pred is None:
        print(f"  {model_name} Metrics: Predictions are None. Skipping calculation.")
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "pearson": np.nan}

    y_true = np.array(y_true).ravel()
    y_pred_arr = np.array(y_pred).ravel()

    if np.all(np.isnan(y_pred_arr)):
        print(f"  {model_name} Metrics: Predictions are all NaN. Skipping calculation.")
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "pearson": np.nan}

    if np.isnan(y_pred_arr).any():
        print(f"  {model_name} Metrics: Predictions contain some NaNs. Skipping calculation.")
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "pearson": np.nan}

    if len(y_true) != len(y_pred_arr):
        print(f"  {model_name} Metrics: y_true and y_pred have different lengths. Skipping calculation.")
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "pearson": np.nan}

    try:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_arr))
        mae = mean_absolute_error(y_true, y_pred_arr)
        r2 = r2_score(y_true, y_pred_arr)

        if np.std(y_true) < 1e-6 or np.std(y_pred_arr) < 1e-6:
            pearson = 1.0 if rmse < 1e-9 else 0.0
        else:
            pearson, _ = pearsonr(y_true, y_pred_arr)
            if np.isnan(pearson):
                pearson = 0.0

        print(f"  {model_name} RMSE: {rmse:.4f}")
        print(f"  {model_name} MAE: {mae:.4f}")
        print(f"  {model_name} R-squared: {r2:.4f}")
        print(f"  {model_name} Pearson Correlation: {pearson:.4f}")
        return {"rmse": rmse, "mae": mae, "r2": r2, "pearson": pearson}
    except Exception as e:
        print(f"  {model_name} Metrics: Error during calculation - {e}. Skipping calculation.")
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "pearson": np.nan}

# -----------------------------
# Define a very simple MLP model
# -----------------------------
class SimpleNN(nn.Module):
    def __init__(self, input_dim_nn, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim_nn, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.network(x)

# Define parameter grids for hyperparameter search for applicable models
param_grid_xgb_student = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6],
    "learning_rate": [0.03, 0.1],
    "subsample": [0.75, 0.9],
    "colsample_bytree": [0.75, 0.9]
}

# Define parameter grid for GRANDE (Reduced for faster testing)
param_grid_grande = {
    'depth': [3, 5],
    'n_estimators': [256, 512],
    'learning_rate_weights': [0.005, 0.01], 
    'learning_rate_index': [0.01],
    'learning_rate_values': [0.01],
    'learning_rate_leaf': [0.005, 0.01],
    'random_seed': [42],
}

# Define parameter grids for other models
param_grid_gradtree = {
    # Expanded hyperparameter grid for GradTree (covering wider search space)
    'depth': [3, 5],
    'learning_rate_index': [0.005, 0.01, 0.05],
    'learning_rate_values': [0.005, 0.01, 0.05],
    'learning_rate_leaf': [0.005, 0.01, 0.05]
}

param_grid_tabnet = {
    'n_d': [8, 16],
    'n_a': [8, 16],
    'n_steps': [3, 5],
    'optimizer_params': [dict(lr=0.005), dict(lr=0.01)]
}

param_grid_simplenn = {
    'hidden_dim': [64, 128],
    'dropout': [0.1, 0.2],
    'lr': [0.001, 0.01]
}

# Helper function to perform GridSearchCV for models supporting sklearn API
def perform_grid_search(estimator, param_grid, X_train_data, y_train_data, cv=3):
    print(f"    Performing GridSearchCV for {estimator.__class__.__name__}...")
    try:
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            verbose=0,
            n_jobs=-1,
        )
        grid_search.fit(X_train_data, y_train_data)
        print(f"    Best parameters found: {grid_search.best_params_}")
        return grid_search.best_params_
    except Exception as e:
        print(f"    Error during GridSearchCV: {e}. Using default parameters.")
        return estimator.get_params()

# Helper function for manual hyperparameter search (for models not compatible with sklearn API)
def manual_hyperparameter_search(model_class, param_grid, X_train_data, y_train_data, X_val_data, y_val_data,
                                model_specific_base_params=None, model_specific_args=None):
    print(f"    Performing manual hyperparameter search for {model_class.__name__}...")
    best_score = float('inf')
    best_params = None
    best_model = None

    # Generate all parameter combinations
    param_combinations = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

    for params in param_combinations:
        try:
            # Create model instance with current parameters
            current_params = model_specific_base_params.copy() if model_specific_base_params is not None else {}
            current_params.update(params)
            
            # Combine params and args for models like GradTree and GRANDE
            model_init_kwargs = {'params': current_params}
            if model_specific_args is not None:
                model_init_kwargs['args'] = model_specific_args

            if model_class.__name__ in ['GRANDE', 'GradTree']:
                model = model_class(params=current_params, args=model_specific_args)
                # Patch instance after creation
                patch_instance_sklearn_tags(model)
            else:
                model = model_class(**current_params)

            # Determine data format for fitting based on model type
            # GradTree and GRANDE expect DataFrames/Series
            X_fit = X_train_data
            y_fit = y_train_data
            X_eval = X_val_data
            y_eval = y_val_data

            # Train model
            if hasattr(model, 'fit'):
                # Pass evaluation set to fit for early stopping if needed
                if (
                    hasattr(model, 'fit') and
                    'X_val' in model.fit.__code__.co_varnames
                ):
                    model.fit(X_fit, y_fit, X_val=X_eval, y_val=y_eval)
                else:
                    model.fit(X_fit, y_fit)

                # Evaluate on validation set (use appropriate data format for prediction)
                y_pred = model.predict(X_eval) # predict on DataFrame/Series

                score = np.sqrt(mean_squared_error(y_eval, y_pred))

                if score < best_score:
                    best_score = score
                    # Store the combined params for best model
                    best_params = current_params
                    best_model = model

        except Exception as e:
            print(f"    Error during training with params {params}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if best_model is None:
        print("    No successful parameter combination found. Using default parameters.")
        # Return default params and None model if search fails
        default_params = model_specific_base_params.copy() if model_specific_base_params is not None else {}
        # If default params also cause error, this might still fail later.
        # Consider adding a flag or returning None/raising error if defaults also fail.
        return default_params, None

    print(f"    Best parameters found: {best_params}")
    print(f"    Best validation score: {best_score:.4f}")
    return best_params, best_model

def main():
    # --- Setup logging to both console and file ---
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    current_time_str = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"new-full_hypersearch_half_teacher_half_holdout_{current_time_str}_log.txt"
    log_filepath = os.path.join(output_dir, log_filename)

    # Initialize metrics storage
    all_metrics = {
        "teacher": {},  # Store teacher model metrics for each dataset
        "students": {   # Store student model metrics for each dataset
            "raw": {},  # Raw training mode
            "distillation": {}  # Knowledge distillation mode
        }
    }

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file_handle = None

    try:
        log_file_handle = open(log_filepath, "w", encoding="utf-8")
        sys.stdout = OutputDuplicator(original_stdout, log_file_handle)
        sys.stderr = OutputDuplicator(original_stderr, log_file_handle)

        # ---------- Main loop: process each dataset ----------
        for dataset_info in DATASETS:
            dataset_name = dataset_info["name"]
            dataset_id = dataset_info["id"]
            print(f"\n{'='*60}")
            print(f"Processing dataset: {dataset_name} (ID: {dataset_id})")
            print(f"{'='*60}")

            # Initialize dataset metrics storage
            all_metrics["teacher"][dataset_name] = {}
            all_metrics["students"]["raw"][dataset_name] = {}
            all_metrics["students"]["distillation"][dataset_name] = {}

            try:
                # ----------------------
                # (1) Data reading and preprocessing
                # ----------------------
                print("\nStep 1: Fetching and preprocessing data...")
                dataset = fetch_openml(data_id=dataset_id, as_frame=True, version="active")
                X = dataset.data
                y = dataset.target

                if X.empty or y.empty:
                    print(f"Dataset {dataset_name} is initially empty. Skipping.")
                    continue

                # Clean column names
                X.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in X.columns]

                # Ensure y is a Series
                if isinstance(y, pd.DataFrame):
                    y = y.iloc[:, 0]

                # Convert target to numeric
                if not pd.api.types.is_numeric_dtype(y):
                    try:
                        if dataset_id == 44133:
                            y = pd.to_numeric(y, errors="raise")
                        elif hasattr(y, "iloc") and isinstance(y.iloc[0], str) and any(c in y.iloc[0] for c in ["$", ","]):
                            y = y.str.replace(r"[$,]", "", regex=True).astype(float)
                        else:
                            y = pd.to_numeric(y, errors="raise")
                    except (ValueError, AttributeError, TypeError, IndexError) as e:
                        print(f"Target column for {dataset_name} is non-numeric ({y.dtype}) and could not be converted: {e}. Skipping dataset.")
                        continue
                y = y.astype(float)

                # Handle missing values in target
                if y.isnull().any():
                    print(f"Target y for {dataset_name} contains NaNs. Removing rows with NaN target.")
                    nan_indices = y.index[y.isnull()]
                    y = y.drop(index=nan_indices)
                    X = X.drop(index=nan_indices)
                    if X.empty or y.empty:
                        print(f"Dataset {dataset_name} became empty after NaN removal from target. Skipping.")
                        continue

                # Reset indices to ensure alignment
                X.reset_index(drop=True, inplace=True)
                y.reset_index(drop=True, inplace=True)

                # Identify feature types
                categorical_features_names = X.select_dtypes(include=["category", "object"]).columns.tolist()
                numerical_features_names = X.select_dtypes(exclude=["category", "object"]).columns.tolist()

                # Handle missing values in numerical features
                if numerical_features_names and X[numerical_features_names].isnull().values.any():
                    num_imputer = SimpleImputer(strategy="mean")
                    X[numerical_features_names] = num_imputer.fit_transform(X[numerical_features_names])

                # Handle missing values in categorical features
                if categorical_features_names:
                    cat_imputer = SimpleImputer(strategy="most_frequent")
                    for col in categorical_features_names:
                        if X[col].isnull().any():
                            X[col] = cat_imputer.fit_transform(X[[col]])[:, 0]
                        X[col] = X[col].astype("category")

                if X.empty:
                    print(f"Dataset {dataset_name} has no features after preprocessing. Skipping.")
                    continue

                # Reintroduce 80/20 train/test split for generalization evaluation
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                if X_train.empty or X_test.empty:
                    print(f"Train or test split resulted in empty data for {dataset_name}. Skipping.")
                    continue

                # Prepare data for GRANDE/GradTree (now based on X_train and X_test from 80/20 split)
                X_train_grande_gradtree = X_train.copy()
                X_test_grande_gradtree = X_test.copy() # Use the actual test set for grande/gradtree evaluation data

                # Process categorical features for GRANDE/GradTree
                # Use X_train_grande_gradtree to identify categories
                categorical_features_names_grande_gradtree = X_train_grande_gradtree.select_dtypes(include=['category', 'object']).columns.tolist()

                for col in categorical_features_names_grande_gradtree:
                    if col in X_train_grande_gradtree.columns:
                        # Ensure category dtype
                        if not pd.api.types.is_categorical_dtype(X_train_grande_gradtree[col]):
                             X_train_grande_gradtree[col] = X_train_grande_gradtree[col].astype('category')

                        # Apply categories from training data to test data
                        if col in X_test_grande_gradtree.columns:
                            # Ensure consistent categories between train and test sets
                            train_categories = X_train_grande_gradtree[col].cat.categories
                            X_test_grande_gradtree[col] = pd.Categorical(X_test_grande_gradtree[col], categories=train_categories)

                        # Convert to numeric codes
                        X_train_grande_gradtree[col] = X_train_grande_gradtree[col].cat.codes
                        if col in X_test_grande_gradtree.columns:
                            X_test_grande_gradtree[col] = X_test_grande_gradtree[col].cat.codes

                # Convert to float32 for GRANDE/GradTree
                X_train_grande_gradtree = X_train_grande_gradtree.astype(np.float32)
                X_test_grande_gradtree = X_test_grande_gradtree.astype(np.float32)

                # Get categorical feature indices (based on original X)
                cat_features_indices = [X.columns.get_loc(col) for col in categorical_features_names]

                # Data split integrity check (now reflects 80/20 split)
                print(f"\nData Split Check:")
                print(f"Training Set: {X_train.shape}, {y_train.shape}")
                print(f"Test Set: {X_test.shape}, {y_test.shape}")

                # Check data overlap (should be 0 between train and test)
                train_indices = set(X_train.index)
                test_indices = set(X_test.index)

                print(f"\nData Overlap Check (Train vs Test): {len(train_indices.intersection(test_indices))}") # Should be 0

                # Check NaN values
                print(f"\nNaN Value Check:")
                print(f"Training Set NaN: {X_train.isna().sum().sum()}, {y_train.isna().sum()}")
                print(f"Test Set NaN: {X_test.isna().sum().sum()}, {y_test.isna().sum()}")

                # Check label distribution across splits
                print(f"\nLabel Distribution Statistics:")
                print("\nTraining Set Labels:")
                print(f"Mean: {y_train.mean():.4f}")
                print(f"Std: {y_train.std():.4f}")
                print(f"Min: {y_train.min():.4f}")
                print(f"Max: {y_train.max():.4f}")

                # Check label distribution shift
                teacher_mean = y_train.mean()
                student_train_mean = y_train.mean()
                student_test_mean = y_test.mean()

                print(f"\nLabel Distribution Shift Check:")
                print(f"Teacher vs Student Train Mean Difference: {abs(teacher_mean - student_train_mean):.4f}")
                print(f"Teacher vs Student Test Mean Difference: {abs(teacher_mean - student_test_mean):.4f}")
                print(f"Student Train vs Test Mean Difference: {abs(student_train_mean - student_test_mean):.4f}")

                # Reset indices
                X_train = X_train.reset_index(drop=True)
                y_train = y_train.reset_index(drop=True)
                X_test = X_test.reset_index(drop=True)
                y_test = y_test.reset_index(drop=True)

                # --- Define parameter grids and args for student models ---

                base_gradtree_params = {
                    'depth': 5,
                    'learning_rate_index': 0.01,
                    'learning_rate_values': 0.01,
                    'learning_rate_leaf': 0.005,
                    'optimizer': 'adam',
                    'cosine_decay_steps': 0,
                    'initializer': 'RandomNormal',
                    'loss': 'mse',
                    'focal_loss': False,
                    'temperature': 0.0,
                    'apply_class_balancing': False,
                    'random_state': 42
                }

                gradtree_args = {
                    'epochs': 100,
                    'early_stopping_epochs': 15,
                    'batch_size': 64,
                    'cat_idx': [],
                    'objective': 'regression',
                    'random_seed': 42,
                    'verbose': 0
                }

                base_grande_params = {
                    'depth': 5,
                    'n_estimators': 1024,
                    'learning_rate_weights': 0.005,
                    'learning_rate_index': 0.01,
                    'learning_rate_values': 0.01,
                    'learning_rate_leaf': 0.01,
                    'optimizer': 'adam',
                    'cosine_decay_steps': 0,
                    'loss': 'mse',
                    'focal_loss': False,
                    'temperature': 0.0,
                    'from_logits': True,
                    'use_class_weights': False,
                    'dropout': 0.0,
                    'selected_variables': 1.0,
                    'data_subset_fraction': 1.0,
                    'random_seed': 42,
                }

                grande_args = {
                    'epochs': 100,
                    'early_stopping_epochs': 15,
                    'batch_size': 64,
                    'cat_idx': [],
                    'objective': 'regression',
                    'verbose': 0
                }

                # Helper function to convert data to numpy arrays
                def prepare_np(X, y):
                    X_ = X.values if hasattr(X, 'values') else X
                    y_ = y.values if hasattr(y, 'values') else y
                    return X_, y_

                # ----------------------
                # (2) Train Parent (Teacher) model and evaluate on Test
                # ----------------------
                print("\nStep 2a: Training Parent Model (Teacher) with Hyperparameter Search...")
                param_grid_parent = {
                    "n_estimators": [100, 250],
                    "max_depth": [4, 6, 9],
                    "learning_rate": [0.03, 0.07, 0.1],
                    "subsample": [0.75, 0.9],
                    "colsample_bytree": [0.75, 0.9],
                }

                if X_train.shape[0] * X_train.shape[1] > 2_000_000:
                    print("Adjusting grid for large dataset for speed.")
                    param_grid_parent = {
                        "n_estimators": [100, 200],
                        "max_depth": [5, 8],
                        "learning_rate": [0.05, 0.1],
                        "subsample": [0.8],
                        "colsample_bytree": [0.8],
                    }

                xgb_parent_search = xg.XGBRegressor(
                    tree_method="hist",
                    enable_categorical=True,
                    random_state=42,
                )
                cv_folds = 2 if X_train.shape[0] > 40000 else 3

                grid_search_parent = GridSearchCV(
                    estimator=xgb_parent_search,
                    param_grid=param_grid_parent,
                    cv=cv_folds,
                    scoring="neg_root_mean_squared_error",
                    verbose=0,
                    n_jobs=-1,
                )
                grid_search_parent.fit(X_train, y_train)
                best_params_parent_from_gs = grid_search_parent.best_params_
                print(f"Best hyperparameters from GridSearchCV: {best_params_parent_from_gs}")

                parent_model = xg.XGBRegressor(
                    tree_method="hist",
                    enable_categorical=True,
                    random_state=42,
                    **best_params_parent_from_gs,
                    early_stopping_rounds=10,
                )

                X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
                    X_train, y_train, test_size=0.15, random_state=42
                )
                if X_train_final.empty or X_val_final.empty:
                    print("Training data too small for validation split. Training parent on full X_train without early stopping.")
                    parent_model.fit(X_train, y_train, verbose=False)
                else:
                    parent_model.fit(
                        X_train_final,
                        y_train_final,
                        eval_set=[(X_val_final, y_val_final)],
                        verbose=False,
                    )

                final_parent_params = parent_model.get_params()
                if hasattr(parent_model, "best_iteration_") and parent_model.best_iteration_ is not None:
                    final_parent_params["n_estimators"] = parent_model.best_iteration_
                else:
                    final_parent_params["n_estimators"] = best_params_parent_from_gs.get(
                        "n_estimators", parent_model.get_params()["n_estimators"]
                    )

                # Get teacher predictions on holdout set for knowledge distillation
                # Correctly predict teacher output on student training data for KD labels
                y_student_train_kd = pd.Series(parent_model.predict(X_train), index=y_train.index)
                
                # Use teacher predictions on test data as KD labels for student test set
                y_student_test_kd = pd.Series(parent_model.predict(X_test), index=y_test.index)

                # The variable y_teacher_pred_on_holdout is already predictions on X_test, so we can keep it if needed elsewhere
                # but for the student test set KD labels, we use y_student_test_kd calculated above.
                
                # Reset indices to avoid any indexing issues in subsequent operations
                X_train = X_train.reset_index(drop=True)
                X_test = X_test.reset_index(drop=True)
                y_train = y_train.reset_index(drop=True)
                y_test = y_test.reset_index(drop=True)
                y_student_train_kd = y_student_train_kd.reset_index(drop=True)
                y_student_test_kd = y_student_test_kd.reset_index(drop=True)

                # KD label alignment check
                print(f"\nKD Label Alignment Check:")
                print(f"Teacher Prediction Label Shape: {parent_model.predict(X_test).shape}")
                print(f"Student Training Set KD Label Shape: {y_student_train_kd.shape}")
                print(f"Student Test Set KD Label Shape: {y_student_test_kd.shape}")

                # Check KD label vs original label correspondence
                print(f"\nKD Label vs Original Label Comparison:")
                print("Student Training Set First 5 Samples:")
                print("Original Labels:", y_train.head())
                print("KD Labels:", y_student_train_kd.head())

                # Check KD label distribution
                print(f"\nKD Label Distribution:")
                print("Student Training Set KD Label Statistics:")
                print(y_student_train_kd.describe())
                print("\nStudent Test Set KD Label Statistics:")
                print(y_student_test_kd.describe())

                # Ensure KD label alignment with data
                y_student_train_kd = y_student_train_kd.reset_index(drop=True)
                y_student_test_kd = y_student_test_kd.reset_index(drop=True)

                print("\nStep 2b: Parent Model Metrics on Test Set:")
                y_pred_parent_test = parent_model.predict(X_test)
                teacher_metrics = calculate_metrics(y_test, y_pred_parent_test, "Parent (Teacher)")
                all_metrics["teacher"][dataset_name] = teacher_metrics

                y_teacher_on_test_np = np.array(y_pred_parent_test).ravel()

                # ----------------------
                # (3) Prepare data for student models
                # ----------------------
                print("\nStep 3: Preparing data for student models...")
                
                # Prepare data for SimpleNN
                X_train_for_nn = X_train.copy()  # Use student training data
                X_test_for_nn = X_test.copy()
                nn_categorical_features = X_train_for_nn.select_dtypes(include=["category"]).columns.tolist()
                nn_numerical_features = X_train_for_nn.select_dtypes(include=np.number).columns.tolist()
                preprocessor_nn = None
                simple_nn_input_dim = 0
                X_train_nn_processed = None
                X_test_nn_processed = None

                transformers_for_nn = []
                if nn_numerical_features:
                    transformers_for_nn.append(("num", StandardScaler(), nn_numerical_features))
                if nn_categorical_features:
                    transformers_for_nn.append(
                        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nn_categorical_features)
                    )

                if not transformers_for_nn:
                    print(f"Warning: Dataset {dataset_name} has no numerical or categorical features for SimpleNN. Skipping NN student.")
                else:
                    preprocessor_nn = ColumnTransformer(transformers=transformers_for_nn, remainder="passthrough")
                    try:
                        X_train_nn_processed = preprocessor_nn.fit_transform(X_train_for_nn)
                        X_test_nn_processed = preprocessor_nn.transform(X_test_for_nn)

                        # Convert to NumPy array and ensure it's 2D
                        X_train_nn_processed_np = np.asarray(X_train_nn_processed)
                        if X_train_nn_processed_np.ndim == 1:
                            X_train_nn_processed_np = X_train_nn_processed_np.reshape(-1, 1)

                        X_test_nn_processed_np = np.asarray(X_test_nn_processed)
                        if X_test_nn_processed_np.ndim == 1:
                            X_test_nn_processed_np = X_test_nn_processed_np.reshape(-1, 1)

                        simple_nn_input_dim = X_train_nn_processed_np.shape[1]
                        if simple_nn_input_dim == 0:
                            print(f"Warning: After preprocessing for SimpleNN, no features remained for dataset {dataset_name}. Skipping NN student.")
                            preprocessor_nn = None
                    except Exception as e_preprocess_nn:
                        print(f"Error during preprocessing for SimpleNN on dataset {dataset_name}: {e_preprocess_nn}. Skipping NN student.")
                        preprocessor_nn = None
                        simple_nn_input_dim = 0

                # Prepare data for TabNet
                X_train_for_tabnet = X_train.copy()  # Use student training data
                X_test_for_tabnet = X_test.copy()
                tabnet_cat_feat_names = X_train_for_tabnet.select_dtypes(include=["category"]).columns.tolist()
                tabnet_num_feat_names = X_train_for_tabnet.select_dtypes(include=np.number).columns.tolist()

                X_train_tabnet_input_np = None
                X_test_tabnet_input_np = None
                cat_idxs_for_tabnet = []
                cat_dims_for_tabnet = []
                can_run_tabnet = False

                if X_train_for_tabnet.empty:
                    print(f"Warning: X_train is empty for TabNet preprocessing on dataset {dataset_name}. TabNet will be skipped.")
                else:
                    processed_numerical_train_np = None
                    processed_numerical_test_np = None
                    if tabnet_num_feat_names:
                        num_pipeline_tabnet = Pipeline([("scaler", StandardScaler())])
                        processed_numerical_train_np = num_pipeline_tabnet.fit_transform(X_train_for_tabnet[tabnet_num_feat_names])
                        processed_numerical_test_np = num_pipeline_tabnet.transform(X_test_for_tabnet[tabnet_num_feat_names])

                    processed_categorical_train_list_np = []
                    processed_categorical_test_list_np = []
                    if tabnet_cat_feat_names:
                        for col_name in tabnet_cat_feat_names:
                            if not pd.api.types.is_categorical_dtype(X_train_for_tabnet[col_name]):
                                X_train_for_tabnet[col_name] = X_train_for_tabnet[col_name].astype("category")
                                X_test_for_tabnet[col_name] = X_test_for_tabnet[col_name].astype("category")

                            if X_train_for_tabnet[col_name].cat.categories.empty and len(X_train_for_tabnet[col_name].unique()) == 0:
                                print(f"Warning: Categorical column {col_name} for TabNet has no categories or only NaNs. Skipping this column.")
                                continue

                            train_encoded_values = X_train_for_tabnet[col_name].cat.codes.values.reshape(-1, 1)
                            test_encoded_values = X_test_for_tabnet[col_name].cat.codes.values.reshape(-1, 1)
                            processed_categorical_train_list_np.append(train_encoded_values)
                            processed_categorical_test_list_np.append(test_encoded_values)
                            
                            current_offset = processed_numerical_train_np.shape[1] if processed_numerical_train_np is not None else 0
                            idx_in_concat = current_offset + len(processed_categorical_train_list_np) - 1
                            cat_idxs_for_tabnet.append(idx_in_concat)
                            cat_dims_for_tabnet.append(len(X_train_for_tabnet[col_name].cat.categories))

                    all_parts_train_for_tabnet = []
                    all_parts_test_for_tabnet = []
                    if processed_numerical_train_np is not None and processed_numerical_train_np.shape[1] > 0:
                        all_parts_train_for_tabnet.append(processed_numerical_train_np)
                        all_parts_test_for_tabnet.append(processed_numerical_test_np)
                    if processed_categorical_train_list_np:
                        concatenated_train_cats_np = np.concatenate(processed_categorical_train_list_np, axis=1)
                        concatenated_test_cats_np = np.concatenate(processed_categorical_test_list_np, axis=1)
                        if concatenated_train_cats_np.shape[1] > 0:
                            all_parts_train_for_tabnet.append(concatenated_train_cats_np)
                            all_parts_test_for_tabnet.append(concatenated_test_cats_np)

                    if all_parts_train_for_tabnet:
                        X_train_tabnet_input_np = np.concatenate(all_parts_train_for_tabnet, axis=1)
                        X_test_tabnet_input_np = np.concatenate(all_parts_test_for_tabnet, axis=1)
                        if X_train_tabnet_input_np.shape[1] > 0:
                            can_run_tabnet = True
                        else:
                            print(f"Warning: After TabNet preprocessing, no features remained for {dataset_name}. Skipping TabNet.")
                    else:
                        print(f"Warning: No numerical or valid categorical features found for TabNet in {dataset_name}. Skipping TabNet.")

                # -------------------------
                # (4) Build configuration list for all student models
                # -------------------------
                student_configs = []

                # (4.1) XGBoost (Same as Parent)
                student_configs.append({
                    "type": "xgboost",
                    "label": "XGBoost (Same as Parent)",
                    "params": final_parent_params.copy(),
                    "is_distillation": False
                })
                # Add distillation version
                student_configs.append({
                    "type": "xgboost",
                    "label": "XGBoost (Same as Parent, Distillation)",
                    "params": final_parent_params.copy(),
                    "is_distillation": True
                })

                # (4.2) GradTree (Actual gradtree.Regressor)
                if _HAS_GRADTREE:
                    try:
                        from GradTree.GradTree import GradTree
                        _HAS_GRADTREE_Regressor = True
                    except ImportError:
                        print("GradTreeRegressor not found. Cannot perform tuning for GradTree.")
                        _HAS_GRADTREE_Regressor = False

                    if _HAS_GRADTREE_Regressor:
                        # Create updated params and args with correct cat_idx
                        current_base_gradtree_params = base_gradtree_params.copy()
                        current_gradtree_args = gradtree_args.copy()
                        current_gradtree_args['cat_idx'] = cat_features_indices

                        # Add raw training version
                        student_configs.append({
                            "type": "gradtree",
                            "label": "GradTree",
                            "params": current_base_gradtree_params.copy(),
                            "args": current_gradtree_args.copy(), # Use the updated args copy
                            "param_grid": param_grid_gradtree,
                            "processed_train_input": X_train_grande_gradtree.copy(),
                            "processed_test_input": X_test_grande_gradtree.copy(),
                            "is_distillation": False
                        })
                        # Add distillation version
                        student_configs.append({
                            "type": "gradtree",
                            "label": "GradTree (Distillation)",
                            "params": current_base_gradtree_params.copy(),
                            "args": current_gradtree_args.copy(), # Use the updated args copy
                            "param_grid": param_grid_gradtree,
                            "processed_train_input": X_train_grande_gradtree.copy(),
                            "processed_test_input": X_test_grande_gradtree.copy(),
                            "is_distillation": True
                        })

                # (4.3) GRANDE
                if _HAS_GRANDE:
                    # Add raw training version
                    student_configs.append({
                        "type": "grande",
                        "label": "GRANDE",
                        "params": base_grande_params.copy(),
                        "args": grande_args.copy(),
                        "param_grid": param_grid_grande,
                        "processed_train_input": X_train_grande_gradtree.copy(),
                        "processed_test_input": X_test_grande_gradtree.copy(),
                        "is_distillation": False
                    })
                    # Add distillation version
                    student_configs.append({
                        "type": "grande",
                        "label": "GRANDE (Distillation)",
                        "params": base_grande_params.copy(),
                        "args": grande_args.copy(),
                        "param_grid": param_grid_grande,
                        "processed_train_input": X_train_grande_gradtree.copy(),
                        "processed_test_input": X_test_grande_gradtree.copy(),
                        "is_distillation": True
                    })

                # (4.4) SimpleNN (MLP)
                if preprocessor_nn and simple_nn_input_dim > 0 and X_train_nn_processed_np is not None:
                    simple_nn_params = {
                        "input_dim_nn": simple_nn_input_dim,
                        "epochs": 60,
                        "batch_size": 32
                    }
                    student_configs.append({
                        "type": "simple_nn",
                        "label": "SimpleNN (MLP)",
                        "params": simple_nn_params,
                        "param_grid": param_grid_simplenn,
                        "preprocessor": preprocessor_nn,
                        "processed_train_input": X_train_nn_processed_np,
                        "processed_test_input": X_test_nn_processed_np,
                        "is_distillation": False
                    })
                    # Add distillation version
                    student_configs.append({
                        "type": "simple_nn",
                        "label": "SimpleNN (MLP, Distillation)",
                        "params": simple_nn_params,
                        "param_grid": param_grid_simplenn,
                        "preprocessor": preprocessor_nn,
                        "processed_train_input": X_train_nn_processed_np,
                        "processed_test_input": X_test_nn_processed_np,
                        "is_distillation": True
                    })

                # (4.5) TabNet
                if can_run_tabnet and X_train_tabnet_input_np is not None and X_train_tabnet_input_np.shape[1] > 0:
                    tabnet_model_params = {
                        "cat_idxs": cat_idxs_for_tabnet,
                        "cat_dims": cat_dims_for_tabnet,
                        "cat_emb_dim": [min(dim // 2 + 1, 50) for dim in cat_dims_for_tabnet] if cat_dims_for_tabnet else 1,
                        "n_d": 8,
                        "n_a": 8,
                        "n_steps": 3,
                        "gamma": 1.3,
                        "lambda_sparse": 1e-3,
                        "optimizer_fn": torch.optim.Adam,
                        "optimizer_params": dict(lr=2e-2),
                        "scheduler_params": {"step_size": 10, "gamma": 0.9},
                        "scheduler_fn": torch.optim.lr_scheduler.StepLR,
                        "mask_type": "sparsemax",
                        "verbose": 0,
                        "seed": 42,
                    }
                    tabnet_fit_params = {
                        "max_epochs": 40,
                        "patience": 7,
                        "batch_size": 256,
                        "virtual_batch_size": 128,
                    }
                    student_configs.append({
                        "type": "tabnet",
                        "label": "TabNet",
                        "params": tabnet_model_params,
                        "fit_params": tabnet_fit_params,
                        "param_grid": param_grid_tabnet,
                        "processed_train_input": X_train_tabnet_input_np,
                        "processed_test_input": X_test_tabnet_input_np,
                        "is_distillation": False
                    })
                    # Add distillation version
                    student_configs.append({
                        "type": "tabnet",
                        "label": "TabNet (Distillation)",
                        "params": tabnet_model_params,
                        "fit_params": tabnet_fit_params,
                        "param_grid": param_grid_tabnet,
                        "processed_train_input": X_train_tabnet_input_np,
                        "processed_test_input": X_test_tabnet_input_np,
                        "is_distillation": True
                    })

                # (4.6) "Poor Student" - Minimal XGBoost version
                params_s4_poor_xgb = {
                    "n_estimators": 70, # These can serve as initial values or be overridden by the grid
                    "max_depth": 4,
                    "learning_rate": 0.06,
                    "subsample": 0.65,
                    "colsample_bytree": 0.75,
                    # Ensure essential XGBoost parameters are included in base params
                    "tree_method": "hist",
                    "enable_categorical": True,
                    "random_state": 42
                }
                student_configs.append({
                    "type": "xgboost",
                    "label": "XGBoost (Poor Student)",
                    "params": params_s4_poor_xgb.copy(), # Use a copy as base parameters
                    "param_grid": param_grid_xgb_student,  # <--- Add parameter grid
                    "is_distillation": False,
                    "use_custom_preproc": False # Keep consistent with other XGBoost student configs
                })
                # Add distillation version
                student_configs.append({
                    "type": "xgboost",
                    "label": "XGBoost (Poor Student, Distillation)",
                    "params": params_s4_poor_xgb.copy(), # Use a copy as base parameters
                    "param_grid": param_grid_xgb_student,  # <--- Add parameter grid
                    "is_distillation": True,
                    "use_custom_preproc": False # Keep consistent with other XGBoost student configs
                })

                # -------------------------
                # (5) Train and evaluate each student model
                # -------------------------
                print(f"\nStep 4: Student Model Training and Evaluation ---")
                for student_config in student_configs:
                    label = student_config["label"]
                    model_type = student_config["type"]
                    is_distillation = student_config.get("is_distillation", False)
                    param_grid = student_config.get("param_grid", None)

                    try:
                        # 只要配置了 param_grid，就做自动超参数搜索；否则默认只跑一组
                        grid = list(ParameterGrid(param_grid)) if param_grid else [student_config["params"]]

                        best_val_metric = float('inf')
                        best_params = None
                        best_pred = None
                        best_model = None

                        for params in grid:
                            current_params = student_config["params"].copy()
                            current_params.update(params)

                            # For models where some grid parameters go into 'args' instead of 'params'
                            current_args = student_config.get("args", {}).copy()
                            if model_type in ["gradtree", "grande"]:
                                 # Update args with parameters from the grid that belong there
                                if "batch_size" in params:
                                    current_args['batch_size'] = params['batch_size']
                                if "early_stopping_epochs" in params:
                                     current_args['early_stopping_epochs'] = params['early_stopping_epochs']

                            try:
                                # === XGBoost ===
                                if model_type == "xgboost":
                                    print(f"\n  Attempting to train {label} with params: {current_params}")
                                    X_train = X_train.copy()
                                    y_train = y_student_train_kd if is_distillation else y_train
                                    # train/val split for early stopping
                                    if len(X_train) > 100:
                                        X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
                                        eval_set = [(X_val, y_val)]
                                        early_stopping_rounds = 10
                                        verbose = False
                                    else:
                                        X_fit, y_fit = X_train, y_train
                                        eval_set = None
                                        early_stopping_rounds = None
                                        verbose = False

                                    model = xg.XGBRegressor(
                                        **current_params,
                                    )
                                    model.fit(X_fit, y_fit, eval_set=eval_set, verbose=verbose)

                                    # For models without param_grid, we just evaluate the one model trained
                                    if param_grid is None:
                                        best_model = model
                                        best_params = current_params.copy()
                                        # Since no hyperparameter search, no validation score to compare, just proceed to test evaluation
                                        break # Exit inner loop as only one config is run
                                    else:
                                        # Evaluate on validation set for hyperparameter tuning
                                        preds_val = model.predict(X_val)
                                        val_rmse = mean_squared_error(y_val, preds_val, squared=False)
                                        print(f"    Validation RMSE: {val_rmse:.4f}")
                                        if val_rmse < best_val_metric:
                                            best_val_metric = val_rmse
                                            best_params = current_params.copy()
                                            best_model = model

                                # === GRANDE ===
                                elif model_type == "grande":
                                    grande_args = student_config["args"]
                                    X_train = student_config["processed_train_input"].copy()
                                    X_test = student_config["processed_test_input"].copy()
                                    y_train = y_student_train_kd if is_distillation else y_train
                                    # train/val split
                                    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
                                    y_fit = y_fit.values if hasattr(y_fit, 'values') else y_fit
                                    y_val = y_val.values if hasattr(y_val, 'values') else y_val
                                    model = GRANDE(params=current_params, args=grande_args)
                                    model.fit(X_train=X_fit, y_train=y_fit, X_val=X_val, y_val=y_val)
                                    preds_val = model.predict(X_val)
                                    val_rmse = mean_squared_error(y_val, preds_val, squared=False)
                                    if val_rmse < best_val_metric:
                                        best_val_metric = val_rmse
                                        best_params = current_params.copy()
                                        best_model = model

                                # === GradTree ===
                                elif model_type == "gradtree":
                                    gradtree_args = student_config["args"]
                                    X_train = student_config["processed_train_input"].copy()
                                    X_test = student_config["processed_test_input"].copy()
                                    y_train = y_student_train_kd if is_distillation else y_train
                                    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
                                    X_fit, y_fit = prepare_np(X_fit, y_fit)
                                    X_val, y_val = prepare_np(X_val, y_val)
                                    model = GradTree(params=current_params, args=current_args)
                                    model.fit(X_train=X_fit, y_train=y_fit, X_val=X_val, y_val=y_val)
                                    preds_val = model.predict(X_val)
                                    val_rmse = mean_squared_error(y_val, preds_val, squared=False)
                                    if val_rmse < best_val_metric:
                                        best_val_metric = val_rmse
                                        best_params = current_params.copy()
                                        best_model = model

                                # === TabNet ===
                                elif model_type == "tabnet":
                                    tn_fit_params = student_config["fit_params"]
                                    tn_model_params = student_config["params"].copy()
                                    tn_model_params.update(params)
                                    X_train_np = student_config["processed_train_input"]
                                    y_train_np = y_student_train_kd if is_distillation else y_train
                                    X_train_np, y_train_np = prepare_np(X_train_np, y_train_np)
                                    y_train_np = y_train_np.reshape(-1, 1)
                                    X_fit, X_val, y_fit, y_val = train_test_split(X_train_np, y_train_np, test_size=0.15, random_state=42)
                                    model = TabNetRegressor(**tn_model_params)
                                    model.fit(
                                        X_train=X_fit, y_train=y_fit,
                                        eval_set=[(X_val, y_val)],
                                        eval_metric=["rmse"],
                                        max_epochs=tn_fit_params["max_epochs"],
                                        patience=tn_fit_params["patience"],
                                        batch_size=tn_fit_params["batch_size"],
                                        virtual_batch_size=tn_fit_params["virtual_batch_size"],
                                        num_workers=0,
                                    )
                                    preds_val = model.predict(X_val).flatten()
                                    val_rmse = mean_squared_error(y_val, preds_val, squared=False)
                                    if val_rmse < best_val_metric:
                                        best_val_metric = val_rmse
                                        best_params = tn_model_params.copy()
                                        best_model = model

                                # === SimpleNN ===
                                elif model_type == "simple_nn":
                                    X_train_nn = student_config["processed_train_input"]
                                    y_train_nn = y_student_train_kd if is_distillation else y_train
                                    
                                    # Ensure data is numpy arrays
                                    X_train_nn_np = np.asarray(X_train_nn)
                                    y_train_nn_np = np.asarray(y_train_nn)
                                    
                                    input_dim = X_train_nn_np.shape[1]
                                    nn_params = student_config["params"].copy()
                                    nn_params.update(params)
                                    epochs = nn_params.get("epochs", 50)
                                    batch_size = nn_params.get("batch_size", 32)
                                    # train/val split
                                    X_fit, X_val, y_fit, y_val = train_test_split(X_train_nn_np, y_train_nn_np, test_size=0.15, random_state=42)
                                    # Ensure X_fit and X_val are numpy arrays for torch
                                    X_fit_np = np.asarray(X_fit)
                                    X_val_np = np.asarray(X_val)

                                    X_fit_tensor = torch.tensor(X_fit_np, dtype=torch.float32)
                                    y_fit_tensor = torch.tensor(y_fit, dtype=torch.float32).unsqueeze(1) # y_fit is already a numpy array
                                    train_dataset = torch.utils.data.TensorDataset(X_fit_tensor, y_fit_tensor)
                                    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                                    
                                    model = SimpleNN(input_dim_nn=input_dim, hidden_dim=nn_params['hidden_dim'], dropout=nn_params['dropout'])
                                    optimizer = torch.optim.Adam(model.parameters(), lr=nn_params['lr'])
                                    criterion = nn.MSELoss()

                                    for epoch in range(epochs):
                                        model.train()
                                        epoch_loss = 0.0
                                        for batch_X, batch_y in train_loader:
                                            optimizer.zero_grad()
                                            out = model(batch_X)
                                            loss = criterion(out, batch_y)
                                            loss.backward()
                                            optimizer.step()
                                            epoch_loss += loss.item() * batch_X.size(0)
                                    
                                    # Evaluate on validation set
                                    model.eval()
                                    with torch.no_grad():
                                        X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32)
                                        preds_val = model(X_val_tensor).numpy().flatten()
                                    val_rmse = mean_squared_error(y_val, preds_val, squared=False)
                                    if val_rmse < best_val_metric:
                                        best_val_metric = val_rmse
                                        best_params = nn_params.copy()
                                        best_model = model

                                else:
                                    print(f"  Unknown model type: {model_type}. Skipping.")
                                    continue # Skip to next param combination or model

                            except Exception as e:
                                print(f"  Params {params} failed for {label}: {e}")
                                import traceback
                                traceback.print_exc()
                                continue

                        
                        if best_model is not None:
                            print(f"  [{label}] Best Parameters: {best_params}")
                            # Prepare test data based on model type
                            if model_type == "simple_nn":
                                X_test_processed = student_config["processed_test_input"]
                                best_model.eval()
                                with torch.no_grad():
                                    X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
                                    y_pred_student_on_test = best_model(X_test_tensor).numpy().flatten()
                            elif model_type in ["grande", "gradtree"]:
                                X_test_processed = student_config["processed_test_input"]
                                y_pred_student_on_test = best_model.predict(X_test_processed)
                            elif model_type == "tabnet":
                                X_test_processed = student_config["processed_test_input"]
                                y_pred_student_on_test = best_model.predict(X_test_processed).flatten()
                            elif model_type == "xgboost":
                                # XGBoost uses original processed data (Pandas DataFrames)
                                X_test_processed = X_test.copy()
                                y_pred_student_on_test = best_model.predict(X_test_processed)
                            else:
                                print(f"  Skipping test prediction for unknown model type: {model_type}")
                                continue # Skip metrics calculation for this model

                            # Store student metrics
                            if is_distillation:
                                # Evaluate ability to mimic teacher (compare with teacher predictions)
                                student_metrics_vs_teacher = calculate_metrics(y_teacher_on_test_np, y_pred_student_on_test, f"Student ({label} vs Teacher)")
                                all_metrics["students"]["distillation"][dataset_name][f"{label} (vs Teacher)"] = student_metrics_vs_teacher

                                # Evaluate performance on original task (compare with true labels)
                                student_metrics_vs_true = calculate_metrics(y_test, y_pred_student_on_test, f"Student ({label} vs True)")
                                all_metrics["students"]["distillation"][dataset_name][f"{label} (vs True)"] = student_metrics_vs_true
                            else:
                                student_metrics = calculate_metrics(y_test, y_pred_student_on_test, f"Student ({label})")
                                all_metrics["students"]["raw"][dataset_name][label] = student_metrics
                        else:
                            print(f"  [{label}] All param combinations failed, skipping.")

                    except Exception as e:
                        print(f"An error occurred while processing {label}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue # Continue to the next student config

            except Exception as e:
                print(f"An error occurred while processing dataset {dataset_name} (ID: {dataset_id}): {e}")
                import traceback
                traceback.print_exc()
                print("Skipping to next dataset.")
                continue

        # Print final comparison of all metrics
        print(f"\n{'='*60}")
        print("Final Metrics Comparison")
        print(f"{'='*60}")

        for dataset_name in all_metrics["teacher"].keys():
            print(f"\nDataset: {dataset_name}")
            print("-" * 40)
            
            # Print teacher metrics
            print("\nTeacher Model Metrics:")
            teacher_metrics = all_metrics["teacher"][dataset_name]
            for metric_name, value in teacher_metrics.items():
                print(f"  {metric_name}: {value:.4f}")

            # Print student metrics (Raw Training)
            print("\nStudent Models (Raw Training) Metrics:")
            raw_metrics = all_metrics["students"]["raw"][dataset_name]
            for model_name, metrics in raw_metrics.items():
                print(f"\n  {model_name}:")
                for metric_name, value in metrics.items():
                    print(f"    {metric_name}: {value:.4f}")

            # Print student metrics (Knowledge Distillation)
            print("\nStudent Models (Knowledge Distillation) Metrics:")
            dist_metrics = all_metrics["students"]["distillation"][dataset_name]
            for model_name, metrics in dist_metrics.items():
                print(f"\n  {model_name}:")
                for metric_name, value in metrics.items():
                    if model_name.startswith('GradTree') or model_name.startswith('GRANDE'):
                        if metric_name == 'r2':
                            try:
                                pass
                            except Exception:
                                pass
                        elif metric_name == 'mae':
                            try:
                                pass
                            except Exception:
                                pass

                    print(f"    {metric_name}: {value:.4f}")

            print("\n" + "="*40)

        # Save metrics to a JSON file
        metrics_filename = f"new-full_optimizer_hypersearch_full_teacher_{current_time_str}_metrics.json"
        metrics_filepath = os.path.join(output_dir, metrics_filename)
        script_name = os.path.basename(__file__)
        all_metrics["script_name"] = script_name
        with open(metrics_filepath, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nMetrics saved to {metrics_filepath}")

    finally:
        # Restore stdout/stderr and close log file
        if sys.stdout is not original_stdout:
            sys.stdout = original_stdout
        if sys.stderr is not original_stderr:
            sys.stderr = original_stderr
        if log_file_handle and not log_file_handle.closed:
            log_file_handle.close()
        original_stdout.write(f"Log saved to {log_filepath}\n")

if __name__ == "__main__":
    main()
