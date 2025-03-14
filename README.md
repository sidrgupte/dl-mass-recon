# **Mass-Reconstruction for DarkLight using Machine Learning**  
[# add later]

## **📌 Table of Contents**
- [Overview](#overview)
<!-- - [Features](#features) -->
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






