# import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
# from skopt import BayesSearchCV
# from skopt.space import Integer, Categorical

import time
import xgboost as xgb
# import optuna
import numpy as np
from sklearn.metrics import mean_squared_error
import json

import matplotlib.pyplot as plt

def read_data(fn):
    data=[]
    with open(fn, 'r') as f:
        for line in f:
            [OUT, EventID, TrackID, ParticleCount1, ParticleCount2, X, Y, dX, dY, E, P, ip, oop, vert_x, vert_y, vert_z] = line.split()
            if int(TrackID) == 1 or int(TrackID) == 2:
                v=[float(P), float(ip), float(oop), float(X), float(Y), float(dX), float(dY)]
                data.append([v[0],v[1],v[2],v[3],v[4],v[5],v[6]])
    return data

# set the device
def set_device():
  device = (
      "cuda"
      if torch.cuda.is_available()
      else "mps"
      if torch.backends.mps.is_available()
      else "cpu"
  )
  print(f"Using {device} device")
  return device

def data(file_path, side, target='all', data_type='train'):
    df = pd.read_csv(file_path, sep='\s+', header=None, names=[
        'OUT', 'EventID', 'TrackID', 'ParticleCount1', 'ParticleCount2', 'X', 'Y', 
        'dX', 'dY', 'E', 'P', 'ip', 'oop', 'vert_x', 'vert_y', 'vert_z'
    ])
    
    if side == 'electron':
        num_outliers = sum(df['TrackID'] != 1)
        df = df[df['TrackID'] == 1]
        print(f"Removed {num_outliers} Outliers from dataset:\t{data_type}!")
    elif side == 'positron':
        num_outliers = sum(df['TrackID'] != 2)
        df = df[df['TrackID'] == 2]
        print(f"Removed {num_outliers} Outliers from dataset:\t{data_type}!!")

    df = df[['P', 'ip', 'oop', 'X', 'Y', 'dX', 'dY']]
    X = df.drop(['P', 'ip', 'oop'], axis=1)

    # target selection
    if target == 'all':
        Y = df[['P', 'ip', 'oop']]
    elif target == 'P':
        Y = df['P']
    elif target == 'ip':
        Y = df['ip']
    elif target == 'oop':
        Y = df['oop']
    else:
        raise ValueError("Invalid target value. Choose from 'all', 'P', 'ip', or 'oop'.")

    X = X.to_numpy()
    Y = Y.to_numpy()

    if data_type == 'test':
        return df, X, Y

    # split the data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    return df, X_train, X_val, Y_train, Y_val

class Analysis:
    def __init__(self, side, scat, data_train, data_test, target, n_jobs = 4, params=None):
        self.side = side
        self.scat = scat
        self.data_train = data_train
        self.data_test = data_test
        self.target = target
        self.n_jobs = n_jobs
        self.params = params

        # load the training and test data
        training_data = self.data_train
        self.df_train, self.X_train, self.X_val, self.Y_train, self.Y_val = data(file_path = training_data, 
                                                                                 target = self.target, 
                                                                                 data_type='train', 
                                                                                 side=self.side)

        test_data = self.data_test
        self.df_test, self.X_test, self.Y_test = data(file_path = test_data, 
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
            save_path = f"{base_path}models/{self.side}/{self.side}_{self.target}_scat_params_dict.json"
        elif self.scat=='n':
            save_path = f"{base_path}models/{self.side}/{self.side}_{self.target}_no_scat_params_dict.json"
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
            save_path = f"{base_path}models/{self.side}/{self.side}_{self.target}_scat.json"
        elif self.scat=='n':
            save_path = f"{base_path}models/{self.side}/{self.side}_{self.target}_no_scat.json"
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

def get_data_path(side, scat, data_path):
    """Returns the training and test dataset paths based on side and scattering condition."""
    
    if side not in ['electron', 'positron']:
        raise ValueError("Invalid 'side' value. Choose 'electron' or 'positron'.")
    
    if scat not in ['y', 'n']:
        raise ValueError("Invalid 'scat' value. Choose 'y' (with scattering) or 'n' (no scattering).")
    
    data_files = {
        'electron': {
            'y': ('ElectronCoords_wide_acp.dat', 'ElectronSort_signal.dat'),
            'n': ('ElectronCoords_no_scat.dat', 'ElectronSort_no_scat.dat')
        },
        'positron': {
            'y': ('PositronCoords_wide_acp.dat', 'PositronSort_signal.dat'),
            'n': ('PositronCoords_no_scat.dat', 'PositronSort_no_scat.dat')
        }
    }

    data_train, data_test = data_files[side][scat]
    data_train = f"{data_path}{data_train}"
    data_test = f"{data_path}{data_test}"

    return data_train, data_test


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