import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler
from torch_geometric_temporal.signal import DynamicHeteroGraphTemporalSignal


def load_and_process_single_measurement(sats_csv_path, receiver_csv_path):
    sats_df_meas = pd.read_csv(sats_csv_path)
    receiver_df_meas = pd.read_csv(receiver_csv_path)
    time_steps_meas = sorted(receiver_df_meas['T_ID'].unique())

    feature_dicts_meas = []
    target_dicts_meas = []
    edge_index_dicts_meas = []
    additional_sids_dicts = []

    for t_local in time_steps_meas:
        rec = receiver_df_meas[receiver_df_meas['T_ID'] == t_local].iloc[0]
        feat_rec = rec[['Lat', 'Lon']].to_numpy().reshape(1, 2)
        targ_rec = rec[['LatDev', 'LonDev']].to_numpy().reshape(1, 2)

        sats_t = sats_df_meas[sats_df_meas['T_ID'] == t_local].sort_values('S_ID')
        feat_sat = sats_t[['SNR', 'az', 'el']].to_numpy()
        s_ids_sat = sats_t['S_ID'].values.astype(np.int64)
        n_sat = feat_sat.shape[0]

        src = np.zeros(n_sat, dtype=int)
        dst = np.arange(n_sat, dtype=int)
        edges = np.vstack([src, dst]) if n_sat > 0 else np.empty((2, 0), dtype=int)
        edges_rev = edges[::-1].copy()

        feature_dicts_meas.append({'receiver': feat_rec, 'satellite': feat_sat})
        target_dicts_meas.append({'receiver': targ_rec})
        edge_index_dicts_meas.append({
            ('receiver', 'to', 'satellite'): edges,
            ('satellite', 'rev_to', 'receiver'): edges_rev
        })
        additional_sids_dicts.append({'satellite_s_ids': s_ids_sat})

    return (
        feature_dicts_meas,
        target_dicts_meas,
        edge_index_dicts_meas,
        [None] * len(time_steps_meas),
        time_steps_meas,
        additional_sids_dicts
    )


def load_all_measurements(measurement_files):
    all_measurements_processed = []
    for m_info in measurement_files:
        features, targets, edges, weights, times, sids = load_and_process_single_measurement(
            m_info["sats"], m_info["receiver"]
        )
        all_measurements_processed.append({
            "id": m_info["id"],
            "features": features,
            "targets": targets,
            "edges": edges,
            "weights": weights,
            "time_steps": times,
            "satellite_s_ids": sids
        })
    return all_measurements_processed


def aggregate_for_normalization(measurements_data):
    agg_rec, agg_sat, agg_targ = [], [], []
    for meas in measurements_data:
        for i in range(len(meas["features"])):
            agg_rec.append(meas["features"][i]['receiver'])
            agg_targ.append(meas["targets"][i]['receiver'])
            fs = meas["features"][i]['satellite']
            if fs.size > 0:
                agg_sat.append(fs)

    return (
        np.vstack(agg_rec)  if agg_rec  else np.empty((0, 2)),
        np.vstack(agg_sat)  if agg_sat  else np.empty((0, 3)),
        np.vstack(agg_targ) if agg_targ else np.empty((0, 2))
    )


def fit_standard_scalers(rec_np, sat_np, targ_np):
    return (
        StandardScaler().fit(rec_np),
        StandardScaler().fit(sat_np),
        StandardScaler().fit(targ_np)
    )


def normalize_with_scalers(measurement_data_list, rec_scaler, sat_scaler, targ_scaler):
    normalized_measurements = []
    for meas_data in measurement_data_list:
        norm_feat_dicts = []
        norm_targ_dicts = []
        norm_sids_list = []

        for i in range(len(meas_data["features"])):
            fr   = meas_data["features"][i]['receiver']
            fs   = meas_data["features"][i]['satellite']
            sids = meas_data["satellite_s_ids"][i]['satellite_s_ids']
            tr   = meas_data["targets"][i]['receiver']

            norm_fr = rec_scaler.transform(fr)
            norm_tr = targ_scaler.transform(tr)
            norm_fs = sat_scaler.transform(fs) if fs.size > 0 else fs.copy()

            norm_feat_dicts.append({'receiver': norm_fr, 'satellite': norm_fs})
            norm_targ_dicts.append({'receiver': norm_tr})
            norm_sids_list.append({'satellite_s_ids': sids.copy()})

        normalized_measurements.append({
            **meas_data,
            "features":        norm_feat_dicts,
            "targets":         norm_targ_dicts,
            "satellite_s_ids": norm_sids_list
        })

    return normalized_measurements


def create_signals(measurements):
    signals = []
    for m in measurements:
        sig = DynamicHeteroGraphTemporalSignal(
            edge_index_dicts=m["edges"],
            edge_weight_dicts=m["weights"],
            feature_dicts=m["features"],
            target_dicts=m["targets"],
            **{"satellite_s_ids": m["satellite_s_ids"]}
        )
        signals.append(sig)
    return signals


class SlidingWindowDataset(Dataset):
    def __init__(self, signal, window_size, stride=1):
        self.signal = signal
        self.window_size = window_size
        self.stride = stride

    def __len__(self):
        return max(0, (self.signal.snapshot_count - self.window_size) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        return [self.signal[t] for t in range(start, start + self.window_size)]


def build_loader(signals, window_size, shuffle, stride=1):
    datasets = []
    for sig in signals:
        ds = SlidingWindowDataset(sig, window_size, stride=stride)
        if len(ds) > 0:
            datasets.append(ds)
    if not datasets:
        return None
    return DataLoader(
        ConcatDataset(datasets),
        batch_size=1,
        shuffle=shuffle,
        collate_fn=lambda batch: batch[0]
    )