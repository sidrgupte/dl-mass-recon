import os, sys
import time
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

# import data_utils:
base_path = os.path.dirname(ipynbname.path())
parent_path = os.path.dirname(base_path)
utils_path = os.path.join(parent_path, "utils")
sys.path.append(utils_path)
import data_utils
importlib.reload(data_utils)
from data_utils import data

class Analysis:
    def __init__(self, side, scat, train_path, test_path, target, n_jobs = 4, params=None):
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

    def run_optimization(self, n_trials, base_path):
        """Runs Optuna hyperparameter tuning if self.optim is True."""
        print("Starting hyperparameter tuning...")
        study = optuna.create_study(direction='minimize')
        
        s = time.time()
        study.optimize(self.objective, n_trials=n_trials) 
        e = time.time()

        self.best_params = study.best_params
        self.best_rmse = study.best_value
        self.optimization_time = e - s

        if self.scat=='y':
            save_path = f"{base_path}/{self.side}_{self.target}_scat_params_dict.json"
        elif self.scat=='n':
            save_path = f"{base_path}/{self.side}_{self.target}_no_scat_params_dict.json"
        with open(save_path, "w") as f:
            json.dump(self.best_params, f, indent=4)

        print("Best Parameters saved to:", save_path)
        print("Best Parameters:", self.best_params)
        print("Best RMSE:", self.best_rmse)
        print(f"Model optimized in : {self.optimization_time:.2f} seconds")
        
    def train_model(self, params=None, verbose=False):
        """Trains an XGBoost model using either optimized or provided parameters."""
        print("Training model with the Selected Parameters...")

        model_params = params if params is not None else self.best_params
        if model_params is None:
            raise ValueError("No parameters provided for training.")
        
        self.model =  xgb.XGBRegressor(**model_params)
        s = time.time()
        self.model.fit(self.X_train, self.Y_train, 
                       eval_set=[(self.X_train, self.Y_train), (self.X_val, self.Y_val)], 
                       verbose=verbose)
        e = time.time()
        self.training_time = e - s
        print(f"Model trained in {self.training_time:.2f} seconds.")
        return self.model

    def save_model(self, base_path):
        if not hasattr(self, 'model'):
            raise ValueError("No trained model found. Train the model before saving.")
       
        if self.scat=='y':
            save_path = f"{base_path}models/{self.side}_{self.target}_scat.json"
        elif self.scat=='n':
            save_path = f"{base_path}models/{self.side}_{self.target}_no_scat.json"
        self.model.save_model(save_path)
        print(f"Model saved successfully at: {save_path}")
        
    def validation(self):
        """Evaluates the model on training and validation sets and calculates RMSE."""
        if not hasattr(self, "model"):
            raise ValueError("No trained model found. Train the model before validation.")
            
        s = time.time()
        self.y_pred = self.model.predict(self.X_val)
        e = time.time()

        self.validation_time = e-s
        print(f"Duration: {self.validation_time}")

        self.y_pred_train = self.model.predict(self.X_train)

        self.rmse_val = np.sqrt(mean_squared_error(self.Y_val, self.y_pred))
        self.rmse_train = np.sqrt(mean_squared_error(self.Y_train, self.y_pred_train))

        print(f"Training RMSE: {self.rmse_train:.4f}")
        print(f"Validation RMSE: {self.rmse_val:.4f}")

        return self.y_pred_train, self.y_pred

    def test(self):
        """Evaluates the model on the test set and calculates RMSE."""
        if not hasattr(self, "model"):
            raise ValueError("No trained model found. Train the model before testing.")
            
        s = time.time()
        self.y_pred_test = self.model.predict(self.X_test)
        e = time.time()
    
        self.test_time = e - s 
        print(f"Test Duration: {self.test_time:.4f} seconds")
    
        self.rmse_test = np.sqrt(mean_squared_error(self.Y_test, self.y_pred_test))
        print(f"Test RMSE: {self.rmse_test:.4f}")
    
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