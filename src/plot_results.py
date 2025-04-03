import os
import sys
import time
import json
import importlib
import argparse
import logging
import warnings
import pickle
import matplotlib.pyplot as plt
import numpy as np

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

if __name__ == "__main__":

    # get the paths:
    base_path = os.path.dirname(os.path.abspath(__file__))
    parent_path = os.path.dirname(base_path)
    raw_data_path = os.path.join(parent_path, 'data', 'raw')
    results_path = f"{parent_path}/results/trained_models" 
    imgs_path = f"{parent_path}/results/images" 

    logging.info(f"parent path: {parent_path}")
    logging.info(f"base path: {base_path}")
    logging.info(f"raw data path: {raw_data_path}")
    logging.info(f"results path: {results_path}")

    os.environ["NUMEXPR_MAX_THREADS"] = '8'


    scat_file = os.path.join(results_path, "results_scat.pkl")
    no_scat_file = os.path.join(results_path, "results_no_scat.pkl")

    # if both exist
    if os.path.exists(scat_file) and os.path.exists(no_scat_file):
        # prompt if both found
        choice = input("Both 'results_scat.pkl' and 'results_no_scat.pkl' are found. Which one would you like to load?\nEnter 'scat' or 'no_scat': ").strip().lower()

        if choice == 'scat':
            file_to_load = scat_file
        elif choice == 'no_scat':
            file_to_load = no_scat_file
        else:
            print("Invalid choice. Please enter either 'scat' or 'no_scat'.")
            exit(1)
    elif os.path.exists(scat_file):
        file_to_load = scat_file
    elif os.path.exists(no_scat_file):
        file_to_load = no_scat_file
    else:
        print("Neither 'results_scat.pkl' nor 'results_no_scat.pkl' were found.")
        exit(1)

    # Load the chosen file
    with open(file_to_load, "rb") as f:
        loaded_results = pickle.load(f)
    print(f"Results successfully loaded from: {file_to_load}")

    # extract data from the saved results:
    true_p, true_ip, true_oop = [], [], []
    pred_p, pred_ip, pred_oop = [], [], []
    for side in ['electron', 'positron']:
        if len(loaded_results[side]) >= 3:  
            pred_p.append(loaded_results[side][0].get('signal_pred', None))
            pred_ip.append(loaded_results[side][1].get('signal_pred', None))
            pred_oop.append(loaded_results[side][2].get('signal_pred', None))

            true_p.append(loaded_results[side][0].get('signal_actual', None))
            true_ip.append(loaded_results[side][1].get('signal_actual', None))
            true_oop.append(loaded_results[side][2].get('signal_actual', None))
        else:
            logging(f"Warning: Not enough data for {side}")

    fig, axs = plt.subplots(3, 2, figsize=(12, 18))  # 3 rows, 2 columns

    # ------------------------------------------------------------------- PLOT #1 -------------------------------------------------------------------
    # p (momentum) 
    y_true_e_p = true_p[0]  
    y_pred_e_p = pred_p[0]  

    axs[0, 0].hist2d(y_true_e_p, y_pred_e_p, bins=150, cmap='plasma')
    axs[0, 0].set_xlabel('True Values')
    axs[0, 0].set_ylabel('Predicted Values')
    axs[0, 0].set_title('Momentum: Electron Signal')
    axs[0, 0].plot([min(y_true_e_p), max(y_true_e_p)], [min(y_true_e_p), max(y_true_e_p)], 'r--', label='Perfect Prediction')
    cbar0 = plt.colorbar(axs[0, 0].collections[0], ax=axs[0, 0])
    cbar0.set_label('Counts')
    axs[0, 0].legend()

    y_true_p = true_p[1]  
    y_pred_p = pred_p[1] 

    axs[0, 1].hist2d(y_true_p, y_pred_p, bins=150, cmap='plasma')
    axs[0, 1].set_xlabel('True Values')
    axs[0, 1].set_ylabel('Predicted Values')
    axs[0, 1].set_title('Momentum: Positron Signal')
    axs[0, 1].plot([min(y_true_p), max(y_true_p)], [min(y_true_p), max(y_true_p)], 'r--', label='Perfect Prediction')
    cbar1 = plt.colorbar(axs[0, 1].collections[0], ax=axs[0, 1])
    cbar1.set_label('Counts')
    axs[0, 1].legend()

    # ip (in-plane angle) 
    y_true_e_ip = true_ip[0]  
    y_pred_e_ip = pred_ip[0] 

    axs[1, 0].hist2d(y_true_e_ip, y_pred_e_ip, bins=150, cmap='plasma')
    axs[1, 0].set_xlabel('True Values')
    axs[1, 0].set_ylabel('Predicted Values')
    axs[1, 0].set_title('In-Plane Angle: Electron Signal')
    axs[1, 0].plot([min(y_true_e_ip), max(y_true_e_ip)], [min(y_true_e_ip), max(y_true_e_ip)], 'r--', label='Perfect Prediction')
    cbar2 = plt.colorbar(axs[1, 0].collections[0], ax=axs[1, 0])
    cbar2.set_label('Counts')
    axs[1, 0].legend()

    y_true_ip = true_ip[1]  
    y_pred_ip = pred_ip[1] 

    axs[1, 1].hist2d(y_true_ip, y_pred_ip, bins=150, cmap='plasma')
    axs[1, 1].set_xlabel('True Values')
    axs[1, 1].set_ylabel('Predicted Values')
    axs[1, 1].set_title('In-Plane Angle: Positron Signal')
    axs[1, 1].plot([min(y_true_ip), max(y_true_ip)], [min(y_true_ip), max(y_true_ip)], 'r--', label='Perfect Prediction')
    cbar3 = plt.colorbar(axs[1, 1].collections[0], ax=axs[1, 1])
    cbar3.set_label('Counts')
    axs[1, 1].legend()

    # oop (out-of-plane angle) 
    y_true_e_oop = true_oop[0]  
    y_pred_e_oop = pred_oop[0]  

    axs[2, 0].hist2d(y_true_e_oop, y_pred_e_oop, bins=150, cmap='plasma')
    axs[2, 0].set_xlabel('True Values')
    axs[2, 0].set_ylabel('Predicted Values')
    axs[2, 0].set_title('Out-of-Plane Angle: Electron Signal')
    axs[2, 0].plot([min(y_true_e_oop), max(y_true_e_oop)], [min(y_true_e_oop), max(y_true_e_oop)], 'r--', label='Perfect Prediction')
    cbar4 = plt.colorbar(axs[2, 0].collections[0], ax=axs[2, 0])
    cbar4.set_label('Counts')
    axs[2, 0].legend()

    y_true_oop = true_oop[1]  
    y_pred_oop = pred_oop[1]  

    axs[2, 1].hist2d(y_true_oop, y_pred_oop, bins=150, cmap='plasma')
    axs[2, 1].set_xlabel('True Values')
    axs[2, 1].set_ylabel('Predicted Values')
    axs[2, 1].set_title('Out-of-Plane Angle: Positron Signal')
    axs[2, 1].plot([min(y_true_oop), max(y_true_oop)], [min(y_true_oop), max(y_true_oop)], 'r--', label='Perfect Prediction')
    cbar5 = plt.colorbar(axs[2, 1].collections[0], ax=axs[2, 1])
    cbar5.set_label('Counts')
    axs[2, 1].legend()

    plt.tight_layout()
    plt.savefig(f"{imgs_path}/heatmap", dpi=300)
    plt.show()


    # ------------------------------------------------------------------- PLOT #2 -------------------------------------------------------------------
    sides = {'electron': 0, 'positron': 1}

    for plot_idx, (category, pred, true) in enumerate([
        ('Momentum', pred_p, true_p),
        ('In-Plane Angle', pred_ip, true_ip),
        ('Out-of-Plane Angle', pred_oop, true_oop)
    ]):
        fig, ax = plt.subplots(2, 2, figsize=(10,8)) 

        for side, side_val in sides.items():
            label = side.capitalize()

            val_pred = pred[side_val]
            val_true = true[side_val]
            bins_true = np.histogram_bin_edges(val_true, bins=200)
            bins_pred = np.histogram_bin_edges(val_pred, bins=200)


            ax[0, 0 if side == 'electron' else 1].hist(val_true, bins=bins_true, label=f"True {label}", histtype="step", linewidth=1.5)
            ax[0, 0 if side == 'electron' else 1].hist(val_pred, bins=bins_pred, label=f"Predicted {label}", histtype="step", linewidth=1.5)
            ax[0, 0 if side == 'electron' else 1].set_title(f"{category}: {label}")
            ax[0, 0 if side == 'electron' else 1].legend()

            diff = val_true - val_pred
            bins = np.histogram_bin_edges(diff, bins=200)

            ax[1, 0 if side == 'electron' else 1].hist(diff, bins=bins, label=f"Residual {label}", histtype="step", linewidth=1.5)
            ax[1, 0 if side == 'electron' else 1].legend()

        fig.suptitle(f'Signal {category} Comparison', fontsize=16)

        save_path = f"signal_comparison_{category.replace(' ', '_').lower()}.png"
        plt.tight_layout()
        plt.savefig(f"{imgs_path}/{save_path}", dpi=300)  
        print(f"Plot for {category} saved as {save_path}")

    plt.tight_layout()
    # plt.subplots_adjust(top=0.5)  
    plt.show()
   
