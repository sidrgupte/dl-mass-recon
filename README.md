# **Mass-Reconstruction for DarkLight using Machine Learning**  
[# add later]

## **📌 Table of Contents**
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
<!-- - [Dataset](#dataset) -->
<!-- - [Results](#results) -->
<!-- - [Contributing](#contributing) -->

---

## **📖 Overview**
This project is a part of the **Mass Reconstruction** plugin for **DarkLight**. Its goal is to **reconstruct the mass at the reaction point** using hit data from GEM detectors.  

Given the hit coordinates **(x, y)** and the angle of impact on the two GEM planes **(dx, dy)**, the model learns to map these values to:  
- **Momentum (p)**  
- **In-plane angle (ip)**  
- **Out-of-plane angle (oop)**  

This mapping is trained using **XGBoost**. Once learned, the model uses these three parameters (**p, ip, oop**) to accurately reconstruct the mass at the reaction point.


---

---
## **⚙️ Installation**
Follow these steps to set up the project on your local machine:  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/sidrgupte/dl-mass-recon.git
cd mass_recon
```

### **2️⃣ Create a Virtual Environment (Optional but Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4️⃣ Verify Installation**
```bash
python --version
pip list  # Shows installed packages
```
---

## **⚙️ Usage**
To run the project, follow these steps:

### 1️⃣ Prepare the Data  
Make sure all the required data files are placed in the `data/raw` directory. This directory should contain the necessary training and test files for the **electron** and **positron** sides, as well as the generator file. The generator file contains TrackID data that will be used for mass reconstruction.

### 2️⃣ Configure the Training Settings  
Before running the scripts, you need to configure the settings for training. The configuration file `src/config/config.json` contains several important parameters:
- **path_train_electron**: Path to the training file for the electron side.
- **path_train_positron**: Path to the training file for the positron side.
- **path_test_electron**: Path to the test file for the electron side.
- **path_test_positron**: Path to the test file for the positron side.
- **path_generator_file**: Path to the generator file, which is used for mass reconstruction later.
- **scat**: Set to `'n'` if physics effects are turned off, or `'y'` if turned on.
- **optimize**: Set to `'y'` if you want to optimize the model's parameters using Optuna, or `'n'` otherwise.
- **n_jobs**: Number of parallel threads to use for training.

### 3️⃣ Train the Models  
Once the configuration file is set up, run the `train.py` script:

```bash
python src/train.py src/config/config.json
```
This script does the following:

- Filters and cleans the training and test data, saving the cleaned data in the `data/filtered` directory. The filtering process retains only data with valid `TrackID`.

- If the optimize flag is set to `'y'`, it uses Optuna to optimize the parameters for each of the models (momentum, in-plane angle, out-of-plane angle).

- Trains six models in total: three for the electron side and three for the positron side. The trained models are saved in the `results/trained_models` directory.

- Tests the models on the provided test data and stores the results in the files `results/results_scat.pkl` (if physics effects are turned on) or `results/results_no_scat.pkl` (if turned off).

### 4️⃣ Plot the Results
After training, run the `plot_results.py` script to visualize the model's performance:
```bash
python src/plot_results.py src/config/config.json
```
This script generates the following plots:
1. A heatmap showing the correlation between the true and predicted values for momentum, in-plane angle, and out-of-plane angle.
2. Histograms comparing the true and predicted values for the above parameters.

The plots are saved in the `results/images directory`.

### 5️⃣ Perform Mass Reconstruction

Finally, run the `mass_recon.py` script to perform mass reconstruction using the trained models and the filtered generator file:
```bash
python src/mass_recon.py src/config/config.json
```
This script takes the filtered generator file, which contains only those `TrackID`s present in the training and test datasets, and performs mass reconstruction. The resulting mass reconstruction plot is saved in the `results/images` directory.
---

## **📂 Project Structure**

<!-- ## **📂 Project Structure**  
The project follows a modular structure to keep code, data, and results organized:  

---
mass_recon/
├── data/                  # Contains datasets
│   ├── raw/               # Unprocessed data directly from detectors
│   ├── filtered/          # Data after applying noise reduction
│   ├── processed/         # Final cleaned-up dataset ready for training
├── results/               # Stores outputs and model predictions
│   ├── images/            # Plots, graphs, and visualizations
│   ├── trained_models/    # Saved models for later inference
├── src/                   # Core source code
│   ├── models/            # Machine learning model scripts
│   ├── utils/             # Utility functions for data processing
├── notebooks/             # Jupyter notebooks for exploration & analysis
├── scripts/               # Standalone scripts for automation (if needed)
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── .gitignore             # Files to exclude from version control
--- -->






