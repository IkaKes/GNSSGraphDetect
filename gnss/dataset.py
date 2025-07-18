import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler
from torch_geometric_temporal.signal import DynamicHeteroGraphTemporalSignal

from .config import window_size

# 1) Single measurement loader:

def load_and_process_single_measurement(sats_csv_path, receiver_csv_path):

    sats_df_meas     = pd.read_csv(sats_csv_path)
    receiver_df_meas = pd.read_csv(receiver_csv_path)
    time_steps_meas = sorted(receiver_df_meas['T_ID'].unique())
    
    feature_dicts_meas    = []
    target_dicts_meas     = []
    edge_index_dicts_meas = []
    additional_sids_dicts = []
    
    for t_local in time_steps_meas:
        rec      = receiver_df_meas[receiver_df_meas['T_ID'] == t_local].iloc[0]
        feat_rec = rec[['Lat', 'Lon']].to_numpy().reshape(1, 2)
        targ_rec = rec[['LatDev', 'LonDev']].to_numpy().reshape(1, 2)
        
        sats_t     = sats_df_meas[sats_df_meas['T_ID'] == t_local].sort_values('S_ID')
        feat_sat   = sats_t[['SNR', 'az', 'el']].to_numpy()
        s_ids_sat  = sats_t['S_ID'].values.astype(np.int64)
        n_sat      = feat_sat.shape[0]
        
#        if n_sat > 0 and n_sat != len(s_ids_sat):
#            s_ids_sat = s_ids_sat[:n_sat]
#       elif n_sat == 0 and len(s_ids_sat) > 0:
#            s_ids_sat = np.array([], dtype=np.int64)
        
        src       = np.zeros(n_sat, dtype=int)
        dst       = np.arange(n_sat, dtype=int)
        edges     = np.vstack([src, dst])
        edges_rev = edges[::-1].copy()
        
        feature_dicts_meas.append({
            'receiver':  feat_rec,
            'satellite': feat_sat
        })
        target_dicts_meas.append({
            'receiver':  targ_rec
        })
        edge_index_dicts_meas.append({
            ('receiver', 'to', 'satellite'):      edges,
            ('satellite', 'rev_to', 'receiver'): edges_rev
        })
        additional_sids_dicts.append({
            'satellite_s_ids': s_ids_sat
        })
    
    edge_weight_dicts_meas = [None] * len(time_steps_meas)
    return (
        feature_dicts_meas,
        target_dicts_meas,
        edge_index_dicts_meas,
        edge_weight_dicts_meas,
        time_steps_meas,
        additional_sids_dicts
    )

# 2) Load and preprocess measurements:

def load_all_measurements(measurement_files):
    all_measurements_processed = []
    for m_info in measurement_files:
        features, targets, edges, weights, times, sids_per_ts = (
            load_and_process_single_measurement(
                m_info["sats"],
                m_info["receiver"]
            )
        )

        all_measurements_processed.append({
            "id":               m_info["id"],
            "features":         features,
            "targets":          targets,
            "edges":            edges,
            "weights":          weights,
            "time_steps":       times,
            "satellite_s_ids":  sids_per_ts
        })

    return all_measurements_processed
    
# 3) Aggregation for normalization:

def aggregate_for_normalization(train_measurements_data):
    agg_train_rec_feats = []
    agg_train_sat_feats = []
    agg_train_targ_rec  = []
    for meas_data in train_measurements_data:
        num_ts = len(meas_data["features"])
        for i in range(num_ts):
            fr = meas_data["features"][i]['receiver']
            agg_train_rec_feats.append(fr)
            fs = meas_data["features"][i]['satellite']
            if fs.size > 0:
                agg_train_sat_feats.append(fs)
            tr = meas_data["targets"][i]['receiver']
            agg_train_targ_rec.append(tr)
    return agg_train_rec_feats, agg_train_sat_feats, agg_train_targ_rec

# 4) Fit StandardScalers:

def fit_standard_scalers(rec_feats_np, sat_feats_np, targ_rec_np):
    rec_scaler  = StandardScaler().fit(rec_feats_np)
    targ_scaler = StandardScaler().fit(targ_rec_np)
    sat_scaler  = StandardScaler().fit(sat_feats_np)
    return rec_scaler, sat_scaler, targ_scaler

# 5) create_signals:

def create_signals(measurements):
    signals = []
    for meas_data in measurements:
        signal = DynamicHeteroGraphTemporalSignal(
            edge_index_dicts   = meas_data["edges"],
            edge_weight_dicts  = meas_data["weights"],
            feature_dicts      = meas_data["features"],
            target_dicts       = meas_data["targets"],
            satellite_s_ids    = meas_data["satellite_s_ids"]
        )
        signals.append(signal)
    return signals

# 6) SlidingWindowDataset & build_loader:

class SlidingWindowDataset(Dataset):
    def __init__(self, signal, window_size, stride=1):
        self.signal = signal
        self.window_size = window_size
        self.stride = stride
    def __len__(self):
        return max(0, (self.signal.snapshot_count - self.window_size) // self.stride + 1)
    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        return [self.signal[t] for t in range(start, end)]

def build_loader(signals, window_size, shuffle, stride=1):
    datasets = []
    for sig in signals:
        ds = SlidingWindowDataset(sig, window_size, stride=stride)
        if len(ds) > 0:
            datasets.append(ds)
    if not datasets:
        return None
    concat = ConcatDataset(datasets)
    return DataLoader(
        concat,
        batch_size=1,
        shuffle=shuffle,
        collate_fn=lambda batch: batch[0]
    )

