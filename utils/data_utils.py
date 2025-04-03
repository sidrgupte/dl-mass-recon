import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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

def get_data(file_path, filter=False):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=[
        'OUT', 'EventID', 'TrackID', 'ParticleCount1', 'ParticleCount2', 'X', 'Y', 
        'dX', 'dY', 'E', 'P', 'ip', 'oop', 'vert_x', 'vert_y', 'vert_z'
    ])

    return df

def filter_data(electron_path, positron_path, type, scat):
    # load the datasets to be cleaned
    electron_data = get_data(electron_path, filter=True)
    positron_data = get_data(positron_path, filter=True)

    len_e1, len_p1 = len(electron_data), len(positron_data)
    print(f"Initial Electron rows: {len_e1}, Initial Positron rows: {len_p1}")
    
    # identify the outliers 1 - wherein TrackIDs do not match
    e_eventid_outliers = electron_data[electron_data['TrackID'] != 1]['EventID'].to_numpy()
    p_eventid_outliers = positron_data[positron_data['TrackID'] != 2]['EventID'].to_numpy()
    bad_eventid = np.concatenate((e_eventid_outliers, p_eventid_outliers)) 
    print(f"TrackID Outliers - Electron: {len(e_eventid_outliers)}, Positron: {len(p_eventid_outliers)}")


    # remove outliers 1
    electron_filtered = electron_data[~electron_data['EventID'].isin(bad_eventid)]
    positron_filtered = positron_data[~positron_data['EventID'].isin(bad_eventid)]
    print(f"After TrackID Filtering - Electron: {len(electron_filtered)}, Positron: {len(positron_filtered)}")

    # common event id's after filtering
    e_eventid = electron_filtered['EventID'].to_numpy()
    p_eventid = positron_filtered['EventID'].to_numpy()
    common_eventid = np.intersect1d(e_eventid, p_eventid)
    print(f"Common EventIDs found: {len(common_eventid)}")

    # EventIDs to be removed due to missing pairs
    unmatched_eventid = np.setdiff1d(np.concatenate((e_eventid, p_eventid)), common_eventid)
    print(f"Unmatched EventIDs after filtering: {len(unmatched_eventid)}")

    # keep events with common EventIDs
    electron_filtered2 = electron_filtered[electron_filtered['EventID'].isin(common_eventid)]
    positron_filtered2 = positron_filtered[positron_filtered['EventID'].isin(common_eventid)]
    print(f"After Common EventID Filtering - Electron: {len(electron_filtered2)}, Positron: {len(positron_filtered2)}")

    len_e2, len_p2 = len(electron_filtered2), len(positron_filtered2)
    print(f"Final Electron: {len_e2}, Final Positron: {len_p2}")

    # save the removed EventIDs
    all_removed_eventid = np.concatenate((bad_eventid, unmatched_eventid)).tolist()

    print(f"Removed {len_e1 - len_e2} outliers from dataset: {electron_path}")
    print(f"Removed {len_p1 - len_p2} outliers from dataset: {positron_path}")

    # save the filtered data (INCOMPLETE)
    parent_dir = os.path.dirname(os.path.dirname(electron_path))  
    filtered_dir = os.path.join(parent_dir, "filtered")  
    os.makedirs(filtered_dir, exist_ok=True)

    filename_e = os.path.basename(electron_path) 
    filename_p = os.path.basename(positron_path) 
    filtered_file_path_e = os.path.join(filtered_dir, f"{filename_e}")
    filtered_file_path_p = os.path.join(filtered_dir, f"{filename_p}")
    electron_filtered2.to_csv(filtered_file_path_e, index=False, sep=' ', header=False)
    positron_filtered2.to_csv(filtered_file_path_p, index=False, sep=' ', header=False)

    print(f"Filtered Electron data saved at: {filtered_file_path_e}")
    print(f"Filtered Positron data saved at: {filtered_file_path_p}")

    removed_eventid_file = os.path.join(filtered_dir, f"{type}_{scat}_removed_eventids.txt")
    with open(removed_eventid_file, "w") as f:
        for event_id in all_removed_eventid:
            f.write(f"{event_id}\n")
    print(f"Saved {len(all_removed_eventid)} removed EventIDs to: {removed_eventid_file}")

    return electron_filtered2, positron_filtered2, all_removed_eventid



def data(file_path, side, target='all', data_type='train'):
    df = get_data(file_path)
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
    data_train = f"{data_path}/{data_train}"
    data_test = f"{data_path}/{data_test}"

    return data_train, data_test


