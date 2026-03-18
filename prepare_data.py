import os
import torch
import yaml
import numpy as np
import random
from pathlib import Path
from gnss import dataset
from gnss.config import PARSED_DATA_DIR

def get_measurement_definitions(base_path):
    measurement_definitions = []
    base_path = Path(base_path)
    print(f"INFO - Scanning folder: {base_path}")
    if not base_path.exists():
        return []
    for subdir in sorted(base_path.iterdir()):
        if subdir.is_dir() and subdir.name.startswith('R'):
            sats_file = subdir / 'sats_data.csv'
            rec_file  = subdir / 'reciever_data.csv'
            if sats_file.exists() and rec_file.exists():
                measurement_definitions.append({
                    "id":       subdir.name,
                    "sats":     str(sats_file),
                    "receiver": str(rec_file),
                })
    return measurement_definitions

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

receiver_name = params['receiver_name']
jamming_type  = params['jamming_type']
jamming_power = params['jamming_power']
seed          = params['seed']

train_val_ratio = params['prepare_data'].get('train_val_ratio', 0.80)
val_split_ratio = params['prepare_data'].get('val_split_ratio', 0.10)

input_dir  = os.path.join(PARSED_DATA_DIR, str(receiver_name), str(jamming_type), str(jamming_power))
output_dir = params.get("output_dir", "processed_data")
os.makedirs(output_dir, exist_ok=True)

measurement_defs = get_measurement_definitions(input_dir)
if not measurement_defs:
    print(f"ERROR - No data found in {input_dir}")
    exit(1)

random.seed(seed)
random.shuffle(measurement_defs)

n = len(measurement_defs)
n_train_val = int(train_val_ratio * n)      
n_val = int(val_split_ratio * n_train_val)  
n_train = n_train_val - n_val               

train_defs = measurement_defs[:n_train]
val_defs   = measurement_defs[n_train : n_train_val]
test_defs  = measurement_defs[n_train_val :]

print(f"Total files: {n}")
print(f"Split: {len(train_defs)} train, {len(val_defs)} val, {len(test_defs)} test")

print("\n Phase 1: Processing train and validation data ")
train_measurements = dataset.load_all_measurements(train_defs)
val_measurements   = dataset.load_all_measurements(val_defs)

print("\n Phase 2: Processing test data ")
if test_defs:
    test_measurements = dataset.load_all_measurements(test_defs)
else:
    print("WARNING - No test data found.")
    test_measurements = []

print("INFO - Computing total number of unique satellites...")
all_sids = set()
for m in train_measurements + val_measurements + test_measurements:
    for ts_sids in m['satellite_s_ids']:
        all_sids.update(ts_sids['satellite_s_ids'].tolist())

num_total_sats = int(max(all_sids) + 1) if all_sids else 20
print(f"INFO - Found {len(all_sids)} unique satellites. num_total_sats = {num_total_sats}")


agg_rec, agg_sat, agg_targ = dataset.aggregate_for_normalization(train_measurements)

if agg_rec.size == 0 or agg_sat.size == 0:
    raise ValueError("Not enough data to fit scalers — check training split.")

rec_scaler_train, sat_scaler_train, targ_scaler_train = dataset.fit_standard_scalers(
    agg_rec, agg_sat, agg_targ
)


train_measurements = dataset.normalize_with_scalers(
    train_measurements, rec_scaler_train, sat_scaler_train, targ_scaler_train)

val_measurements = dataset.normalize_with_scalers(
    val_measurements, rec_scaler_train, sat_scaler_train, targ_scaler_train)

train_signals = dataset.create_signals(train_measurements)
val_signals   = dataset.create_signals(val_measurements)

if test_measurements:
    agg_rec_t, agg_sat_t, agg_targ_t = dataset.aggregate_for_normalization(test_measurements)
    rec_scaler_test, sat_scaler_test, targ_scaler_test = dataset.fit_standard_scalers(
        agg_rec_t, agg_sat_t, agg_targ_t)
    test_measurements = dataset.normalize_with_scalers(
        test_measurements, rec_scaler_test, sat_scaler_test, targ_scaler_test)
    test_signals = dataset.create_signals(test_measurements)
else:
    test_signals    = []
    rec_scaler_test = None

print(f"\nSaving processed data to: {output_dir}")

torch.save(train_signals,  os.path.join(output_dir, 'train_graphs.pt'))
torch.save(val_signals,    os.path.join(output_dir, 'val_graphs.pt'))
torch.save(test_signals,   os.path.join(output_dir, 'test_graphs.pt'))
torch.save(num_total_sats, os.path.join(output_dir, 'num_total_sats.pt'))

torch.save(rec_scaler_train,  os.path.join(output_dir, 'rec_scaler_train.pt'))
torch.save(sat_scaler_train,  os.path.join(output_dir, 'sat_scaler_train.pt'))
torch.save(targ_scaler_train, os.path.join(output_dir, 'targ_scaler_train.pt'))

if rec_scaler_test:
    torch.save(rec_scaler_test,  os.path.join(output_dir, 'rec_scaler_test.pt'))
    torch.save(sat_scaler_test,  os.path.join(output_dir, 'sat_scaler_test.pt'))
    torch.save(targ_scaler_test, os.path.join(output_dir, 'targ_scaler_test.pt'))

print("\nData preparation complete.")