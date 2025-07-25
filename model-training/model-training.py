"""
LightGBM Model Training Pipeline for Obesity Prediction

This script implements a complete machine learning pipeline for training
a LightGBM classifier to predict obesity levels based on demographic and
lifestyle factors.

Key Features:
- Hyperparameter optimization using Optuna
- Cross-validation for robust model evaluation
- Comprehensive logging and error handling
- Model persistence with metadata tracking
- Configurable training parameters

Usage:
    python model-training.py

Requirements:
    - Preprocessed data in 'data/preprocessed' directory
    - Python packages: lightgbm, optuna, scikit-learn, pandas, numpy

Authors: DSI Obesity Prediction Team
Date: 2025
"""

import os
import pandas as pd
import lightgbm as lgb
import numpy as np
import optuna
import pickle
import logging
import yaml
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from datetime import datetime

def load_config(config_path: str = "config.yml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration YAML file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        # Use print here since logging might not be configured yet
        print(f"✅ Configuration loaded from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        print(f"❌ Invalid YAML configuration: {e}")
        raise

def read_data(data_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load the most recent preprocessed data from the specified path"""
    try:
        folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        if not folders:
            raise ValueError(f"No preprocessed data folders found in {data_path}")
            
        latest_folder = sorted(folders)[-1]
        latest_data_path = os.path.join(data_path, latest_folder)
        logging.info(f"Loading data from the most recent folder: '{latest_data_path}'")

        # Load the scaled data sets
        X_train = pd.read_csv(os.path.join(latest_data_path, 'X_train_scaled.csv'))
        y_train = pd.read_csv(os.path.join(latest_data_path, 'y_train_scaled.csv')).squeeze()
        
        logging.info(f"Data loaded successfully. Training with {len(X_train)} samples.")
        logging.info(f"Features: {X_train.shape[1]}, Classes: {len(y_train.unique())}")

        return X_train, y_train
    except Exception as e:
        logging.error(f"❌ Error loading data: {e}")
        raise

def fit_model(X_train: pd.DataFrame, y_train: pd.Series, params: dict) -> lgb.LGBMClassifier:
    """Fit a LightGBM model with given parameters"""
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model

def set_general_model_params(params: dict, yaml_config: dict) -> dict:
    """Set additional parameters for the LightGBM model"""
    lgbm_config = yaml_config['training']['lightgbm']
    params.update({
        'verbosity': lgbm_config.get('verbosity', -1),
        'random_state': yaml_config.get('random_state', None),
        'objective': lgbm_config.get('objective', 'multiclass'),
        'num_class': lgbm_config.get('num_class', 7)
    })

    gpu_enabled = yaml_config['training']['gpu'].get('enabled', False)
    if gpu_enabled:
        params.update({
            'gpu_device_id': yaml_config['training']['gpu'].get('id', 0),
            'gpu_platform_id': yaml_config['training']['gpu'].get('platform_id', 0),
        })
    return params

def set_specific_model_params(trial: optuna.Trial, yaml_config: dict) -> dict:
    """
    Define hyperparameter search space for Optuna optimization from YAML config.
    
    Args:
        trial: Optuna trial object for suggesting hyperparameters
        yaml_config: YAML configuration dictionary
        
    Returns:
        Dictionary of hyperparameters for this trial
    """
    params = {}
    lgbm_config = yaml_config['training']['lightgbm']
    
    for param_name, param_config in lgbm_config.items():
        # Skip non-parameter entries like verbosity, objective, num_class
        if not isinstance(param_config, dict) or 'enabled' not in param_config:
            continue
            
        # Skip disabled parameters
        if not param_config.get('enabled', False):
            continue
            
        param_type = param_config.get('type', 'float')
        
        if param_type == 'int':
            params[param_name] = trial.suggest_int(
                param_name,
                param_config['minimum'],
                param_config['maximum'],
                step=param_config.get('step', 1)
            )
        elif param_type == 'float':
            params[param_name] = trial.suggest_float(
                param_name,
                param_config['minimum'],
                param_config['maximum'],
                step=param_config.get('step', 0.01)
            )
        elif param_type == 'str':
            params[param_name] = trial.suggest_categorical(
                param_name,
                param_config['values']
            )
        else:
            logging.warning(f"Unknown parameter type '{param_type}' for {param_name}, skipping")
    
    return params

def get_all_model_params(trial: optuna.Trial, yaml_config: dict) -> dict:
    """Combine general and specific model parameters"""
    params = set_specific_model_params(trial, yaml_config)
    params = set_general_model_params(params, yaml_config)
    return params

def objective(X_train: pd.DataFrame, y_train: pd.Series, trial: optuna.Trial, yaml_config: dict) -> float:
    """
    Optuna objective function for LightGBM hyperparameter optimization.
    
    Args:
        X_train: Training features DataFrame
        y_train: Training labels Series
        trial: Optuna trial object for suggesting hyperparameters
        yaml_config: YAML configuration dictionary
        
    Returns:
        Mean cross-validation accuracy score
        
    Raises:
        Exception: Logs any errors during model training and re-raises
    """
    try:
        # Get hyperparameters for this trial
        params = get_all_model_params(trial, yaml_config)
        
        # Log trial parameters for debugging
        logging.info(f"Trial {trial.number}: Testing parameters {params}")

        # Get random state from config
        random_state = yaml_config.get('random_state', 42)
        
        # Stratified cross-validation for balanced splits
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            try:
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx] 
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                # Train model on fold
                model = fit_model(X_tr, y_tr, params)
                
                # Get validation score
                val_pred = model.predict(X_val)
                val_score = accuracy_score(y_val, val_pred)
                scores.append(val_score)
                
                logging.debug(f"Trial {trial.number}, Fold {fold_idx + 1}: Accuracy = {val_score:.4f}")
                
            except Exception as fold_error:
                logging.error(f"Trial {trial.number}, Fold {fold_idx + 1} failed: {str(fold_error)}")
                raise
        
        mean_accuracy = np.mean(scores)
        std_accuracy = np.std(scores)
        
        # Log results
        logging.info(f"Trial {trial.number}: CV Accuracy = {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        
        return mean_accuracy
        
    except Exception as e:
        logging.error(f"Trial {trial.number} failed with error: {str(e)}")
        # Re-raise to let Optuna handle the failed trial
        raise

def save_model_metadata(
    model_path: str, 
    best_params: dict, 
    best_value: float, 
    train_accuracy: float, 
    timestamp: str, 
    training_samples: int,
    n_trials: int
) -> str:
    """
    Save model training metadata to pickle file.
    
    Args:
        model_path: Directory path to save metadata
        best_params: Best hyperparameters from optimization
        best_value: Best cross-validation score
        train_accuracy: Training accuracy of final model
        timestamp: Training timestamp
        training_samples: Number of training samples
        
    Returns:
        Path to saved metadata file
        
    Raises:
        Exception: If metadata saving fails
    """
    try:
        metadata = {
            'best_params': best_params,
            'best_cv_accuracy': best_value,
            'training_accuracy': train_accuracy,
            'overfitting': train_accuracy - best_value,
            'timestamp': timestamp,
            'training_samples': training_samples,
            'model_type': 'LightGBM',
            'n_trials': n_trials
        }

        metadata_path = os.path.join(model_path, f"lgbm_metadata_{timestamp}.pkl")
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logging.info(f"📋 Model metadata saved to: {metadata_path}")
        return metadata_path
        
    except Exception as e:
        logging.error(f"Failed to save model metadata: {str(e)}")
        raise

def save_model(model, model_path: str, model_name: str) -> str:
    """
    Save trained model to pickle file.
    
    Args:
        model: Trained LightGBM model
        model_path: Directory path to save model
        model_name: Name of the model file
        
    Returns:
        Full path to saved model file
        
    Raises:
        Exception: If model saving fails
    """
    try:
        full_model_path = os.path.join(model_path, model_name)
        
        with open(full_model_path, 'wb') as f:
            pickle.dump(model, f)

        # Verify the file was created and has content
        if os.path.exists(full_model_path) and os.path.getsize(full_model_path) > 0:
            logging.info(f"✅ Model saved successfully to: {full_model_path}")
            logging.info(f"📊 Model file size: {os.path.getsize(full_model_path) / 1024:.2f} KB")
        else:
            raise Exception("Model file was not created or is empty")
        
        return full_model_path
        
    except Exception as e:
        logging.error(f"Failed to save model: {str(e)}")
        raise


def filter_features(X_train: pd.DataFrame, yaml_config: dict) -> pd.DataFrame:
    """
    Filter X_train to only include features specified in the config file.
    
    Args:
        X_train: Original training features DataFrame
        yaml_config: YAML configuration dictionary containing feature selection
        
    Returns:
        Filtered DataFrame with only selected features
        
    Raises:
        ValueError: If no valid features are found
    """
    selected_features = yaml_config['training']['features']['names']
    logging.info(f"🔍 Filtering features based on config.yml")
    logging.info(f"Available features in dataset: {list(X_train.columns)}")
    logging.info(f"Selected features from config: {selected_features}")
    
    # Check if all selected features exist in the dataset
    missing_features = [f for f in selected_features if f not in X_train.columns]
    if missing_features:
        logging.warning(f"⚠️ Missing features in dataset: {missing_features}")
        # Remove missing features from selection
        selected_features = [f for f in selected_features if f in X_train.columns]
        logging.info(f"Updated selected features (removed missing): {selected_features}")
    
    # Validate that we have at least one feature
    if not selected_features:
        raise ValueError("No valid features found in dataset from config selection")
    
    # Filter X_train to only include selected features
    X_train_filtered = X_train[selected_features].copy()
    logging.info(f"✅ Features filtered - Original: {X_train.shape[1]} features, Selected: {X_train_filtered.shape[1]} features")
    logging.info(f"Final dataset shape: {X_train_filtered.shape}")
    
    return X_train_filtered

def log_optimization_parameters(yaml_config: dict) -> None:
    """
    Log which parameters will be optimized and which are disabled.
    
    Args:
        yaml_config: YAML configuration dictionary
    """
    lgbm_config = yaml_config['training']['lightgbm']
    enabled_params = []
    disabled_params = []
    
    for param_name, param_config in lgbm_config.items():
        if isinstance(param_config, dict) and 'enabled' in param_config:
            if param_config.get('enabled', False):
                param_type = param_config.get('type', 'unknown')
                if param_type == 'str':
                    range_info = f"values: {param_config.get('values', [])}"
                else:
                    range_info = f"{param_config.get('minimum', 'N/A')}-{param_config.get('maximum', 'N/A')}"
                enabled_params.append(f"{param_name} ({param_type}: {range_info})")
            else:
                disabled_params.append(param_name)
    
    logging.info(f"🔧 Parameters to optimize ({len(enabled_params)}): {', '.join(enabled_params)}")
    if disabled_params:
        logging.info(f"❌ Disabled parameters ({len(disabled_params)}): {', '.join(disabled_params)}")

def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the training process"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )

def validate_config(config: dict) -> None:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = ['data_path', 'model_path', 'n_trials', 'log_level', 'random_state']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate data path exists
    if not os.path.exists(config['data_path']):
        raise ValueError(f"Data path does not exist: {config['data_path']}")
    
    # Validate n_trials is positive
    if config['n_trials'] <= 0:
        raise ValueError(f"n_trials must be positive, got: {config['n_trials']}")
    
    # Validate log level
    valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if config['log_level'].upper() not in valid_log_levels:
        raise ValueError(f"Invalid log level: {config['log_level']}")
    
    logging.info("✅ Configuration validation passed")

def main():
    """
    Main training pipeline for LightGBM obesity prediction model.
    
    This function orchestrates the entire training process:
    1. Sets up logging and configuration
    2. Loads preprocessed data
    3. Optimizes hyperparameters with Optuna
    4. Trains final model with best parameters
    5. Saves model and metadata
    """
    
    try:
        # Load configuration from YAML file
        yaml_config = load_config("config.yml")
        
        # Extract data_path from YAML config and set up other parameters
        CONFIG = {
            'data_path': yaml_config['data_path'],
            'model_path': yaml_config['training']['model_path'],
            'n_trials': yaml_config['training']['optuna']['n_trials'],
            'log_level': yaml_config.get('log_level', 'INFO'),
            'random_state': yaml_config.get('random_state', None)
        }
        
        # Setup logging first
        setup_logging(CONFIG['log_level'])
        
        # Validate configuration
        validate_config(CONFIG)
        
        logging.info("🚀 Starting LightGBM model training pipeline")
        logging.info(f"Configuration: {CONFIG}")
        logging.info(f"Using data_path from config.yml: {CONFIG['data_path']}")
        
        # Load data into variables
        preprocessed_dir = os.path.join('.', CONFIG['data_path'])
        logging.info(f"Loading data from: {preprocessed_dir}")
        X_train, y_train = read_data(preprocessed_dir)
        
        logging.info(f"✅ Data loaded successfully - Shape: {X_train.shape}, Classes: {y_train.nunique()}")
        logging.info(f"Class distribution:\n{y_train.value_counts().sort_index()}")

        # Filter features based on config.yml
        X_train = filter_features(X_train, yaml_config)

        # Log which parameters will be optimized
        log_optimization_parameters(yaml_config)

        # Set up Optuna study with better configuration
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )
        
        logging.info(f"🔧 Starting hyperparameter optimization with {CONFIG['n_trials']} trials")
        study.optimize(
            lambda trial: objective(X_train, y_train, trial, yaml_config), 
            n_trials=CONFIG['n_trials'],
            show_progress_bar=True
        )
        
        logging.info(f"🎯 Optimization completed!")
        logging.info(f"Best parameters: {study.best_params}")
        logging.info(f"Best CV accuracy: {study.best_value:.4f}")

        # Create a Unique Name for Output Model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"lgbm_model_{timestamp}.pkl"
        logging.info(f"Output model name: {model_name}")

        # Get best parameters and add required parameters
        best_params = study.best_params.copy()
        best_params = set_general_model_params(best_params, yaml_config)
        logging.info(f"Final parameters for training: {best_params}")
        
        # Create and train the final model
        logging.info("🏋️ Training final model with best parameters...")
        final_model = fit_model(X_train, y_train, best_params)

        # Create output directory if it doesn't exist
        os.makedirs(CONFIG['model_path'], exist_ok=True)

        # Save the model to pickle file
        save_model(final_model, CONFIG['model_path'], model_name)

        # Verify the model was saved correctly
        train_accuracy = accuracy_score(y_train, final_model.predict(X_train))
        overfitting = train_accuracy - study.best_value
        
        logging.info(f"📊 Final model training accuracy: {train_accuracy:.4f}")
        logging.info(f"📊 Cross-validation accuracy: {study.best_value:.4f}")
        logging.info(f"📊 Overfitting (Train - CV): {overfitting:.4f}")
        
        # Save model metadata
        save_model_metadata(
            CONFIG['model_path'], 
            best_params, 
            study.best_value, 
            train_accuracy, 
            timestamp, 
            len(X_train),
            CONFIG['n_trials']
        )
        
        logging.info("✅ Training pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"❌ Training pipeline failed with error: {str(e)}")
        logging.error("Stack trace:", exc_info=True)
        raise

if __name__ == "__main__":
    main()
