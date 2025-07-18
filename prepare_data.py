import os
import torch
import yaml
import numpy as np
import random
from pathlib import Path
from gnss import dataset
from gnss.features import normalize_with_scalers
from gnss.config import PARSED_DATA_DIR

# (1) Load parameters:
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Read exp group parameters
receiver_name = params['receiver_name']
jamming_type = params['jamming_type']
jamming_power = params['jamming_power']
seed = params['seed']
window_size = params['prepare_data']['window_size']
stride = params['prepare_data']['stride']
train_ratio = params['prepare_data'].get('train_ratio', 0.7)
val_ratio = params['prepare_data'].get('val_ratio', 0.15)
assert train_ratio + val_ratio < 1.0, "train_ratio + val_ratio must be < 1.0"

# (2) Set up input/output dirs:

input_dir = os.path.join(PARSED_DATA_DIR, str(receiver_name), str(jamming_type), str(jamming_power))
output_dir = params.get("output_dir")
if not output_dir:
    raise ValueError("No global 'output_dir' found in params.yaml!")
os.makedirs(output_dir, exist_ok=True)
print(f" INFO - Using input directory: {input_dir}")
print(f" INFO - Saving outputs to: {output_dir}")


# (2) Discover measurement files (one per ts):

def get_measurement_definitions(base_path):
    measurement_definitions = []
    base_path = Path(base_path)
    print(f" DEBUG - Scanning for measurement folders in: {base_path}")
    for subdir in sorted(base_path.iterdir()):
        if subdir.is_dir() and subdir.name.startswith('R'):
            sats_file = subdir / 'sats_data.csv'
            rec_file = subdir / 'reciever_data.csv'
            if sats_file.exists() and rec_file.exists():
                measurement_definitions.append({
                    "id": subdir.name,
                    "sats": str(sats_file),
                    "receiver": str(rec_file),
                })
                print(f" DEBUG - Found measurement: {subdir.name} | sats: {sats_file} | rec: {rec_file}")
            else:
                print(f" WARNING - Missing sats or receiver file in {subdir}")
    print(f" INFO -Found {len(measurement_definitions)} measurement pairs in {base_path}")
    return measurement_definitions

measurement_defs = get_measurement_definitions(input_dir)
if not measurement_defs:
    print(f" WARNING -  No valid measurement pairs found in {input_dir}.")
    exit(0)

# (3) Split:

random.seed(seed)
random.shuffle(measurement_defs)

n = len(measurement_defs)
n_train = int(train_ratio * n)
n_val = int(val_ratio * n)

train_defs = measurement_defs[:n_train]
val_defs = measurement_defs[n_train : n_train + n_val]
test_defs = measurement_defs[n_train + n_val :]
print(f"Split: {len(train_defs)} train, {len(val_defs)} val, {len(test_defs)} test time series")
print("Train IDs:", [d['id'] for d in train_defs])
print("Val IDs:", [d['id'] for d in val_defs])
print("Test IDs:", [d['id'] for d in test_defs])


#########################################################################

print("\n--- FAZA 1: Obrada Trening i Validacijskih podataka ---")

print(" INFO - Učitavanje trening i validacijskih mjerenja...")
train_measurements = dataset.load_all_measurements(train_defs)
val_measurements = dataset.load_all_measurements(val_defs)

print(" INFO - Fitanje 'train' scalera na trening podacima")
agg_train_rec_feats, agg_train_sat_feats, agg_train_targ_rec = dataset.aggregate_for_normalization(train_measurements)
if not all(len(x) > 0 for x in [agg_train_rec_feats, agg_train_sat_feats, agg_train_targ_rec]):
    raise ValueError("Nema podataka za fitanje trening scalera.")

agg_train_rec_feats_2d = np.vstack(agg_train_rec_feats)
agg_train_sat_feats_2d = np.vstack(agg_train_sat_feats) if agg_train_sat_feats else np.empty((0,3))
agg_train_targ_rec_2d = np.vstack(agg_train_targ_rec)

rec_scaler_train, sat_scaler_train, targ_scaler_train = dataset.fit_standard_scalers(
    agg_train_rec_feats_2d, agg_train_sat_feats_2d, agg_train_targ_rec_2d
)

print("INFO - Normalizacija trening i validacijskih podataka...")
normalized_train_measurements = normalize_with_scalers(train_measurements, rec_scaler_train, sat_scaler_train, targ_scaler_train)
normalized_val_measurements = normalize_with_scalers(val_measurements, rec_scaler_train, sat_scaler_train, targ_scaler_train)

train_signals = dataset.create_signals(normalized_train_measurements)
val_signals = dataset.create_signals(normalized_val_measurements)

####################################################################

print("\n--- FAZA 2: Obrada Testnih podataka ---")

print(" INFO - Učitavanje testnih mjerenja...")
test_measurements = dataset.load_all_measurements(test_defs)

if test_measurements:
    print(" INFO - Fitanje 'test' scalera na testnim podacima...")
    agg_test_rec_feats, agg_test_sat_feats, agg_test_targ_rec = dataset.aggregate_for_normalization(test_measurements)
    if not all(len(x) > 0 for x in [agg_test_rec_feats, agg_test_sat_feats, agg_test_targ_rec]):
        raise ValueError("Nema podataka za fitanje test scalera.")

    agg_test_rec_feats_2d = np.vstack(agg_test_rec_feats)
    agg_test_sat_feats_2d = np.vstack(agg_test_sat_feats) if agg_test_sat_feats else np.empty((0,3))
    agg_test_targ_rec_2d = np.vstack(agg_test_targ_rec)

    rec_scaler_test, sat_scaler_test, targ_scaler_test = dataset.fit_standard_scalers(
        agg_test_rec_feats_2d, agg_test_sat_feats_2d, agg_test_targ_rec_2d
    )

    print(" INFO - Normalizacija testnih podataka...")
    normalized_test_measurements = normalize_with_scalers(test_measurements, rec_scaler_test, sat_scaler_test, targ_scaler_test)
    test_signals = dataset.create_signals(normalized_test_measurements)
else:
    print(" WARNIGN - Nema testnih podataka za obradu. Testni skup će biti prazan.")
    test_signals = []
    rec_scaler_test, sat_scaler_test, targ_scaler_test = None, None, None


print(f"\nINFO - Spremanje svih podataka i scalera u: {output_dir}")

torch.save(train_signals, os.path.join(output_dir, 'train_graphs.pt'))
torch.save(val_signals, os.path.join(output_dir, 'val_graphs.pt'))
torch.save(test_signals, os.path.join(output_dir, 'test_graphs.pt'))

print("INFO - Spremanje 'train' seta scalera...")
torch.save(rec_scaler_train, os.path.join(output_dir, 'rec_scaler_train.pt'))
torch.save(sat_scaler_train, os.path.join(output_dir, 'sat_scaler_train.pt'))
torch.save(targ_scaler_train, os.path.join(output_dir, 'targ_scaler_train.pt'))

if rec_scaler_test:
    print("INFO - Spremanje 'test' seta scalera...")
    torch.save(rec_scaler_test, os.path.join(output_dir, 'rec_scaler_test.pt'))
    torch.save(sat_scaler_test, os.path.join(output_dir, 'sat_scaler_test.pt'))
    torch.save(targ_scaler_test, os.path.join(output_dir, 'targ_scaler_test.pt'))

print("\nPriprema podataka je završena.")


'''
# (4) Load and process train measurements only for scaler fitting
train_measurements = dataset.load_all_measurements(train_defs, window_size)
val_measurements = dataset.load_all_measurements(val_defs, window_size)
test_measurements = dataset.load_all_measurements(test_defs, window_size)

agg_train_rec_feats, agg_train_sat_feats, agg_train_targ_rec = dataset.aggregate_for_normalization(train_measurements)
if len(agg_train_rec_feats) == 0 or len(agg_train_sat_feats) == 0 or len(agg_train_targ_rec) == 0:
    print(f"[WARNING] No valid features found for training in {input_dir}. Skipping.")
    exit(0)

try:
    # Receiver features: stack to [N, F]
    agg_train_rec_feats_2d = np.vstack(agg_train_rec_feats)
except Exception as e:
    print(f"WARNING- Could not stack receiver features: {e}. Skipping.")
    exit(0)

# Satellite features: stack all non-empty, check all have isti broj feature-a
non_empty_sat_feats = [fs for fs in agg_train_sat_feats if fs.size > 0]
if non_empty_sat_feats:
    n_features = non_empty_sat_feats[0].shape[1]
    if not all(fs.shape[1] == n_features for fs in non_empty_sat_feats):
        print(f"WARNING - Inconsistent satellite feature dimensions (columns) in {input_dir}. Skipping.")
        exit(0)
    agg_train_sat_feats_2d = np.vstack(non_empty_sat_feats)
else:
    n_features = agg_train_rec_feats_2d.shape[1] if agg_train_rec_feats_2d.size > 0 else 0
    agg_train_sat_feats_2d = np.empty((0, n_features))

try:
    agg_train_targ_rec_2d = np.vstack(agg_train_targ_rec)
except Exception as e:
    print(f"WARNING - Could not stack target features: {e}. Skipping.")
    exit(0)

# (5) Fit scalers on train set

rec_scaler, sat_scaler, targ_scaler = dataset.fit_standard_scalers(
    agg_train_rec_feats_2d, agg_train_sat_feats_2d, agg_train_targ_rec_2d)

# (6) Normalize sve skupove koristeci train scalere
normalized_train_measurements = normalize_with_scalers(train_measurements, rec_scaler, sat_scaler, targ_scaler)
normalized_val_measurements = normalize_with_scalers(val_measurements, rec_scaler, sat_scaler, targ_scaler)
normalized_test_measurements = normalize_with_scalers(test_measurements, rec_scaler, sat_scaler, targ_scaler)

# (7) Convert to graph objects (signals)
train_signals = dataset.create_signals(normalized_train_measurements)
val_signals = dataset.create_signals(normalized_val_measurements)
test_signals = dataset.create_signals(normalized_test_measurements)

# (8) Save outputs
print(f"Saving {len(train_signals)} train, {len(val_signals)} val, {len(test_signals)} test graphs to {output_dir}")
torch.save(train_signals, os.path.join(output_dir, 'train_graphs.pt'))
torch.save(val_signals, os.path.join(output_dir, 'val_graphs.pt'))
torch.save(test_signals, os.path.join(output_dir, 'test_graphs.pt'))
torch.save(rec_scaler, os.path.join(output_dir, 'rec_scaler.pt'))
torch.save(sat_scaler, os.path.join(output_dir, 'sat_scaler.pt'))
torch.save(targ_scaler, os.path.join(output_dir, 'targ_scaler.pt'))

print("Data preparation complete.")
'''