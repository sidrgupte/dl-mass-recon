import os
import sys
import time
import json
import importlib
import argparse
import logging
import warnings
import pickle

# login configuration
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

console_handler = logging.StreamHandler() 
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler('script.log')  
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler])


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Load training and testing data paths from a JSON file")
    parser.add_argument('config_file', type=str, help="Path to the JSON configuration file")
    return parser.parse_args()

def load_config(config_file):
    """Function to load the configuration from a JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    logging.info("=" * 100)
    logging.info(f"Script started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 100 + "\n")

    args = parse_arguments()
    try:
        config = load_config(args.config_file)
        logging.info("Configuration file loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        sys.exit(1)

    # extract params from the config file:
    print(f"{'-' * 40} Extracting params from the Config file {'-' * 40}")
    path_train_electron = config["path_train_electron"]
    path_test_electron = config["path_test_electron"]
    path_train_positron = config["path_train_positron"]
    path_test_positron = config["path_test_positron"]
    scat = config["scat"]
    optimize = config["optimize"]
    n_jobs = config["n_jobs"]
    verbose = config["verbose"]

    os.environ["NUMEXPR_MAX_THREADS"] = str(n_jobs)

    base_path = os.path.dirname(os.path.abspath(__file__))
    parent_path = os.path.dirname(base_path)
    raw_data_path = os.path.join(parent_path, 'data', 'raw')

    logging.info(f"parent path: {parent_path}")
    logging.info(f"base path: {base_path}")
    logging.info(f"raw data path: {raw_data_path}")

    # import  ml_model.py and utils.py
    ml_model_path = os.path.join(parent_path, "src", "models")
    utils_path = os.path.join(parent_path, "utils")
    sys.path.append(ml_model_path)
    sys.path.append(utils_path)
    
    try:
        import ml_model, data_utils
        importlib.reload(ml_model)
        importlib.reload(data_utils)

        from ml_model import Analysis
        logging.info("Successfully loaded ml_model.py")

        from data_utils import filter_data
        logging.info("Successfully loaded utils.py")

    except Exception as e:
        logging.error(f"Error importing modules: {e}")
        sys.exit(1)

    logging.info(f"Training Data Path (Electron): {path_train_electron}")
    logging.info(f"Testing Data Path (Electron): {path_test_electron}")
    logging.info(f"Training Data Path (Positron): {path_train_positron}")
    logging.info(f"Testing Data Path (Positron): {path_test_positron}\n")

    # filter the data on the bases of EventID:
    print(f"{'-' * 40} Filtering Training Data {'-' * 40}")
    _, _, _ = filter_data(electron_path=path_train_electron,
                          positron_path=path_train_positron, 
                          type='train', 
                          scat=scat)
    logging.info("Training data filtered.\n")

    print(f"{'-' * 40} Filtering Test Data {'-' * 40}")
    _, _, _ = filter_data(electron_path=path_test_electron,
                          positron_path=path_test_positron, 
                          type='test', 
                          scat=scat)
    logging.info("Test data filtered.\n")

    filtered_data_path = os.path.join(parent_path, 'data', 'filtered')
    filename_train_electron = os.path.basename(path_train_electron)
    filename_test_electron = os.path.basename(path_test_electron)
    filename_train_positron = os.path.basename(path_train_positron)
    filename_test_positron = os.path.basename(path_test_positron)

    path_train_electron_filtered = os.path.join(filtered_data_path, filename_train_electron)
    path_test_electron_filtered = os.path.join(filtered_data_path, filename_test_electron)
    path_train_positron_filtered = os.path.join(filtered_data_path, filename_train_positron)
    path_test_positron_filtered = os.path.join(filtered_data_path, filename_test_positron)

    """ ------------------------------------------------------------- Optimization Begins ------------------------------------------------------------- """
    if optimize=="y":
        print(f"{'-' * 40} Optimization Begins {'-' * 40}")

        optim_save_path = os.path.join(parent_path, 'src', 'config')

        ### ELECTRON Optimization
        side = 'electron'
        optim_save_path_electron = os.path.join(optim_save_path, 'electron')
        
        for target in ['P', 'ip', 'oop']:
            electron = Analysis(side=side, 
                                scat=scat,
                                train_path = path_train_electron_filtered,
                                test_path = path_test_electron_filtered,
                                target = target,
                                n_jobs = n_jobs)
            electron.run_optimization(n_trials=50,
                                      path=optim_save_path_electron)

        ### POSITRON Optimization
        side = 'positron'
        optim_save_path_positron = os.path.join(optim_save_path, 'positron')

        for target in ['P', 'ip', 'oop']:
            positron = Analysis(side=side, 
                                scat=scat,
                                train_path = path_train_positron_filtered,
                                test_path = path_test_positron_filtered,
                                target = target,
                                n_jobs = n_jobs)
            positron.run_optimization(n_trials=50,
                                      path=optim_save_path_positron)

        print(f"{'~' * 40} Optimization Ends {'~' * 40}")
        """ ------------------------------------------------------------- Optimization Ends ------------------------------------------------------------- """
    else:
        print(f"{'-' * 40} Optimization: No {'-' * 40}")
        logging.info(f"Using optimum parameters stored at {os.path.join(parent_path, 'src', 'config')}\n")

    
    """ ------------------------------------------------------------------ Training Begins ------------------------------------------------------------------ """
    results = {"electron": [], "positron": []}
    for side in ["electron", "positron"]:

        print(f"{'-' * 40} Training Begins: {side.capitalize()} {'-' * 40}")

        for target in ['P', 'ip', 'oop']:
            logging.info(f"Processing: {target} for {side.capitalize()}")

            current_data = {}

            model = Analysis(side=side, scat=scat,
                             train_path=path_train_electron_filtered if side == "electron" else path_train_positron_filtered,
                             test_path=path_test_electron_filtered if side == "electron" else path_test_positron_filtered,
                             target=target, n_jobs=10)

            # load the best params
            path_optim_params = f"{parent_path}/src/config/{side}/{side}_{target}_{'scat' if scat=='y' else 'no_scat'}_params_dict.json"
            with open(path_optim_params, "r") as f:
                best_params = json.load(f)
            current_data["best_params"] = best_params

            # train the model
            trained_model = model.train_model(best_params)
            current_data["trained_model"] = trained_model

            # save the model
            model.save_model(f"{parent_path}/results/trained_models")

            # validation
            train_pred, val_pred = model.validation()
            current_data["train_actual"] = model.Y_train
            current_data["train_pred"] = train_pred
            current_data["val_pred"] = val_pred
            current_data["val_actual"] = model.Y_val

            # testing
            test_pred = model.test()
            current_data["signal_pred"] = test_pred
            current_data["signal_actual"] = model.Y_test

            logging.info(f"Completed Processing: {target} for {side.capitalize()}\n")
            results[side].append(current_data)
    
        print(f"{'~' * 40} Training Ends: {side.capitalize()} {'~' * 40}\n")
        
    # save the models + dictionaries
    save_path = f"{parent_path}/results/trained_models" 
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, f"results_{'scat' if scat == 'y' else 'no_scat'}.pkl"), "wb") as f:
        pickle.dump(results, f)
    logging.info(f"Results successfully saved to: {save_path}/results_{'scat' if scat == 'y' else 'no_scat'}.pkl")



    logging.info("#" * 100)
    logging.info(f"Script execution completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("#" * 100 + "\n")