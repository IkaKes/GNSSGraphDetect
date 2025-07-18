import numpy as np

def normalize_with_scalers(measurement_data_list, rec_scaler, sat_scaler, targ_scaler):
    normalized_measurements = []
    for meas_data in measurement_data_list:
        norm_feat_dicts = []
        norm_targ_dicts = []
        norm_sids_list  = []
        num_ts = len(meas_data["features"])
        for i in range(num_ts):
            fr   = meas_data["features"][i]['receiver']
            fs   = meas_data["features"][i]['satellite']
            sids = meas_data["satellite_s_ids"][i]['satellite_s_ids']
            tr   = meas_data["targets"][i]['receiver']
            norm_fr = rec_scaler.transform(fr)
            norm_tr = targ_scaler.transform(tr)
            if fs.size > 0:
                norm_fs = sat_scaler.transform(fs)
            else:
                norm_fs = fs.copy()
            norm_feat_dicts.append({
                'receiver':  norm_fr,
                'satellite': norm_fs
            })
            norm_targ_dicts.append({
                'receiver':  norm_tr
            })
            norm_sids_list.append({'satellite_s_ids': sids.copy()})
        new_meas = {
            **meas_data,
            "features":        norm_feat_dicts,
            "targets":         norm_targ_dicts,
            "satellite_s_ids": norm_sids_list
        }
        normalized_measurements.append(new_meas)
    return normalized_measurements
