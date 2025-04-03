import os
import sys
import json
import argparse
import logging
import warnings
import pickle
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp
from math import pi, sqrt, tan
from scipy import optimize
import pandas as pd

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
    parser = argparse.ArgumentParser(description="Load training and testing data paths from a JSON file")
    parser.add_argument('config_file', type=str, help="Path to the JSON configuration file")
    return parser.parse_args()

def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

if __name__ == "__main__":

    args = parse_arguments()
    try:
        config = load_config(args.config_file)
        logging.info("Configuration file loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        sys.exit(1)

    generator_file = config["path_generator_file"]
    scat = config["scat"]

    base_path = os.path.dirname(os.path.abspath(__file__))
    parent_path = os.path.dirname(base_path)
    raw_data_path = os.path.join(parent_path, 'data', 'raw')
    results_path = f"{parent_path}/results/trained_models" 
    imgs_path = f"{parent_path}/results/images" 

    logging.info(f"parent path: {parent_path}")
    logging.info(f"base path: {base_path}")
    logging.info(f"raw data path: {raw_data_path}")
    logging.info(f"results path: {results_path}")
    logging.info(f"generator file: {generator_file}")

    os.environ["NUMEXPR_MAX_THREADS"] = '8'

    with open(os.path.join(results_path, f"results_{'scat' if scat == 'y' else 'no_scat'}.pkl"), "rb") as f:
        loaded_results = pickle.load(f)
    print("\nResults successfully loaded!")

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
            logging.warning(f"Warning: Not enough data for {side}")

    beam_size = "(0.,0.)mm"
    signal_data = pd.read_csv(generator_file, sep='\s+', header=None, names=['OUT', 'num', 'p1', 'px1', 'py1', 'pz1', 'p2', 'px2', 'py2', 'pz2', 'weight', 'posrand'])    
    print(f"Length of Signal Data: {len(signal_data)}")

    def gauss(x, *p):
        A, mu, sigma = p
        return A * exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    d2r = pi / 180

    p1_file = pred_p[0]
    ip1_file = pred_ip[0]
    oop1_file = pred_oop[0]

    p2_file = pred_p[1]
    ip2_file = pred_ip[1]
    oop2_file = pred_oop[1]

    print(f"\nChecking if the length of Electron and Proton data for p, ip, and oop are equal...")
    print(f"Momentum:\t\t{len(p1_file) == len(p2_file)}")
    print(f"In-Plane Angle:\t\t{len(p1_file) == len(p2_file)}")
    print(f"Out-of-Plane Angle:\t{len(p1_file) == len(p2_file)}\n")

    def calcm():
        p1 = np.array(p1_file)
        ip1 = np.array(ip1_file)  
        oop1 = np.array(oop1_file)
        
        p2 = np.array(p2_file)
        ip2 = np.array(ip2_file) 
        oop2 = np.array(oop2_file)  

        d2r = np.pi / 180  

        so = np.tan(oop1 * d2r)
        si = np.tan(ip1 * d2r)
        l = np.sqrt(1 + si**2 + so**2)
        s1 = np.array([si / l, so / l, 1 / l])

        so = np.tan(oop2 * d2r)
        si = np.tan(ip2 * d2r)
        l = np.sqrt(1 + si**2 + so**2)
        s2 = np.array([si / l, so / l, 1 / l])

        a1, a2 = 36, -20
        a1_rad, a2_rad = a1 * d2r, a2 * d2r

        s1r = np.array([
            s1[0] * np.cos(a1_rad) + s1[2] * np.sin(a1_rad),
            s1[1],
            -s1[0] * np.sin(a1_rad) + s1[2] * np.cos(a1_rad)
        ])
        
        s2r = np.array([
            s2[0] * np.cos(a2_rad) + s2[2] * np.sin(a2_rad),
            s2[1],
            -s2[0] * np.sin(a2_rad) + s2[2] * np.cos(a2_rad)
        ])

        m = np.sqrt(p1**2 + 0.511**2) + np.sqrt(p2**2 + 0.511**2) 
        p = p1 * s1r + p2 * s2r 
        m2 = np.sqrt(m**2 - np.sum(p**2, axis=0))  

        return m2

    res = calcm()
    w = signal_data['weight']

    range_min, range_max = min(res), max(res)

    mass_hist, bin_edges = np.histogram(res, weights=w, bins=500, range=[range_min, range_max])
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    initial_guess = [max(mass_hist), np.mean(res), np.std(res)]
    coeff, var_matrix = optimize.curve_fit(gauss, bin_centres, mass_hist, p0=initial_guess)
    A_fit, mu_fit, sigma_fit = coeff

    x_fit = np.linspace(range_min, range_max, 1000)
    y_fit = gauss(x_fit, A_fit, mu_fit, sigma_fit)

    plt.hist(res, weights=w, bins=bin_edges, range=[range_min, range_max], histtype='step', linewidth=1.5, density=False, label="Recon. Data")
    plt.plot(x_fit, y_fit, label=f"$\\mu={mu_fit:.1f}$ MeV/c$^2$\n$\\sigma={sigma_fit*1000:.1f}$ keV/c$^2$", color='orange', linewidth=2)

    plt.ylabel("Sum of Weights")
    plt.title("Photon Mass Reconstruction")
    plt.xlabel("Mass [MeV/c$^2$]")
    plt.xlim(12.8, 13.2)
    plt.savefig(f"{imgs_path}/mass_recon", dpi=300)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{imgs_path}/mass_recon.pdf", dpi=300)
    plt.show()
