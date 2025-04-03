import os, sys
import time
import logging
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import ipynbname
import importlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Set up logger (ensure this is done at the start of your file, if not already)
logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,  # or DEBUG depending on your preference
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler("training.log")  # Save to log file
    ]
)

# import data_utils:
base_path = os.path.dirname(os.path.abspath(__file__)) 
parent_path = os.path.dirname(base_path)
utils_path = os.path.join(parent_path, "utils")
sys.path.append(utils_path)
import data_utils
importlib.reload(data_utils)
from data_utils import data
from typing import Literal

class Analysis:
    def __init__(self, 
                train_path: str, 
                test_path: str, 
                side: Literal["electron", "positron"],
                target: Literal['P', 'ip', 'oop', 'all'], 
                n_jobs = 4, 
                scat: Literal['n', 'y'] = 'y',
                params=None):

        self._validate_inputs(side, scat, target)

        self.side = side
        self.scat = scat
        self.train_path = train_path
        self.test_path = test_path
        self.target = target
        self.n_jobs = n_jobs
        self.params = params

        # load the training and test data
        self.df_train, self.X_train, self.X_val, self.Y_train, self.Y_val = data(file_path = self.train_path, 
                                                                                 target = self.target, 
                                                                                 data_type='train', 
                                                                                 side=self.side)

        self.df_test, self.X_test, self.Y_test = data(file_path = self.test_path, 
                                                      target = self.target, 
                                                      data_type='test', 
                                                      side=self.side)

    def _validate_inputs(self, side: str, scat: str, target: str):
        if side not in ["electron", "positron"]:
            raise ValueError('side must be either "electron" or "positron"')
        
        if scat not in ['n', 'y']:
            raise ValueError('scat must be either "n" or "y"')
        
        if target not in ['P', 'ip', 'oop', 'all']:
            raise ValueError('target must be one of "P", "ip", "oop", or "all"')

    def objective(self, trial):
        """Objective function for Optuna parameter tuning."""
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'n_jobs': self.n_jobs,
            'early_stopping_rounds': 10
        }

        # train XGBoost model
        model = xgb.XGBRegressor(**params)
        model.fit(self.X_train, self.Y_train, 
                  eval_set=[(self.X_train, self.Y_train), (self.X_val, self.Y_val)], 
                  verbose=False)
        
        # predict & calculate RMSE
        y_pred = model.predict(self.X_val)
        rmse = np.sqrt(mean_squared_error(self.Y_val, y_pred)) 

        return rmse

    def run_optimization(self, n_trials, path):
        """Runs Optuna hyperparameter tuning if self.optim is True."""
        logging.info("Starting hyperparameter tuning...")
        study = optuna.create_study(direction='minimize')
        
        s = time.time()
        study.optimize(self.objective, n_trials=n_trials) 
        e = time.time()

        self.best_params = study.best_params
        self.best_rmse = study.best_value
        self.optimization_time = e - s

        save_path = f"{path}/{self.side}/{self.side}_{self.target}_{'scat' if self.scat=='y' else 'no_scat'}_params_dict.json"
        with open(save_path, "w") as f:
            json.dump(self.best_params, f, indent=4)

        logging.info(f"Best Parameters: {self.best_params}")
        logging.info(f"Best Parameters saved to: {save_path}")
        logging.info(f"Best RMSE: {self.best_rmse}")
        logging.info(f"Model optimized in {self.optimization_time:.2f} seconds")
        
    # def train_model(self, params=None, verbose=False):
    #     """Trains an XGBoost model using either optimized or provided parameters."""
    #     print("Training model with the Selected Parameters...")

    #     model_params = params if params is not None else self.best_params
    #     if model_params is None:
    #         raise ValueError("No parameters provided for training.")
        
    #     self.model =  xgb.XGBRegressor(**model_params)
    #     s = time.time()
    #     self.model.fit(self.X_train, self.Y_train, 
    #                    eval_set=[(self.X_train, self.Y_train), (self.X_val, self.Y_val)], 
    #                    verbose=verbose)
    #     e = time.time()
    #     self.training_time = e - s
    #     print(f"Model trained in {self.training_time:.2f} seconds.")
    #     return self.model

    def train_model(self, params=None, verbose=False):
        """Trains an XGBoost model using either optimized or provided parameters."""
        logger.info("Training model with the Selected Parameters...")

        model_params = params if params is not None else self.best_params
        if model_params is None:
            logger.error("No parameters provided for training.")
            raise ValueError("No parameters provided for training.")
        
        # Log the model parameters being used for training (only log if verbose logging is needed)
        logger.debug(f"Using parameters for training: {model_params}")

        # Create the XGBoost model
        self.model = xgb.XGBRegressor(**model_params)

        # Start training
        logger.info("Training started...")
        s = time.time()
        self.model.fit(self.X_train, self.Y_train, 
                    eval_set=[(self.X_train, self.Y_train), (self.X_val, self.Y_val)], 
                    verbose=verbose)
        e = time.time()
        
        # Log the training time
        self.training_time = e - s
        logger.info(f"Model trained in {self.training_time:.2f} seconds.")

        # Optionally, log the training progress (if verbose is True in fit function)
        if verbose:
            logger.info("Training progress printed on console.")

        return self.model

    def save_model(self, path):
        if not hasattr(self, 'model'):
            logging.error("No trained model found. Train the model before saving.")
            raise ValueError("No trained model found. Train the model before saving.")
       
        save_path = f"{path}/{self.side}/{self.side}_{self.target}_{'scat' if self.scat=='y' else 'no_scat'}.bin"
        self.model.save_model(save_path)
        logging.info(f"Model saved successfully at: {save_path}")
        
    def validation(self):
        """Evaluates the model on training and validation sets and calculates RMSE."""
        if not hasattr(self, "model"):
            logging.error("No trained model found. Train the model before validation.")
            raise ValueError("No trained model found.")
            
        s = time.time()
        self.y_pred = self.model.predict(self.X_val)
        e = time.time()
        self.validation_time = e - s

        self.y_pred_train = self.model.predict(self.X_train)
        self.rmse_val = np.sqrt(mean_squared_error(self.Y_val, self.y_pred))
        self.rmse_train = np.sqrt(mean_squared_error(self.Y_train, self.y_pred_train))

        logging.info(f"Validation completed in {self.validation_time:.2f} seconds.")
        logging.info(f"Training RMSE: {self.rmse_train:.4f}, Validation RMSE: {self.rmse_val:.4f}")

        return self.y_pred_train, self.y_pred

    def test(self):
        """Evaluates the model on the test set and calculates RMSE."""
        if not hasattr(self, "model"):
            logging.error("No trained model found. Train the model before testing.")
            raise ValueError("No trained model found.")
            
        s = time.time()
        self.y_pred_test = self.model.predict(self.X_test)
        e = time.time()
        self.testing_time = e - s
        
        self.rmse_test = np.sqrt(mean_squared_error(self.Y_test, self.y_pred_test))

        logging.info(f"Testing completed in {self.testing_time:.2f} seconds.")
        logging.info(f"Test RMSE: {self.rmse_test:.4f}")
    
        return self.y_pred_test 


def plot_histograms(data, title):
    """
    Plots histograms for actual vs predicted values and their residuals.
    
    Parameters:
    - data: Dictionary containing 'train_actual', 'train_pred', 
      'val_actual', 'val_pred', 'test_actual', 'test_pred'.
    - title: Title for the entire figure.
    """
    fig, axes = plt.subplots(2, 3, figsize=(10, 5))  # Slightly larger figure for better visibility

    sets = ["train", "val", "signal"]
    
    # First row: Actual vs Predicted
    for i, subset in enumerate(sets):
        val = data[f"{subset}_actual"]
        val_pred = data[f"{subset}_pred"]
        bins = np.histogram_bin_edges(val, bins=200)
        
        axes[0, i].hist(val, bins=bins, label="Actual", histtype="step", linewidth=1.5)
        axes[0, i].hist(val_pred, bins=bins, label="Pred", histtype="step", linewidth=1.5)
        axes[0, i].legend()
        axes[0, i].set_title(f"{subset.capitalize()}")
    
    # Second row: Residuals (Actual - Predicted)
    for i, subset in enumerate(sets):
        residuals = data[f"{subset}_actual"] - data[f"{subset}_pred"]
        bins = np.histogram_bin_edges(residuals, bins=200)
        
        axes[1, i].hist(residuals, bins=bins, histtype="step", linewidth=1.5)
        axes[1, i].set_title(f"Residuals ({subset.capitalize()})")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{title}.pdf", dpi=300)
    plt.savefig(f"{title}", dpi=300)
    plt.show()