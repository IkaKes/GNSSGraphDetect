import torch
import torch.nn as nn
from torch_geometric_temporal.nn.hetero import HeteroGCLSTM


class JaGuard(nn.Module):
    def __init__(self, in_channels_dict, hidden_dim, metadata, num_total_sats, dropout_rate=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_total_sats = num_total_sats

        self.gclstm = HeteroGCLSTM(
            in_channels_dict=in_channels_dict,
            out_channels=hidden_dim,
            metadata=metadata
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_out = nn.Linear(hidden_dim, 2)

    def forward(self, window_snapshots, device):
        # Initialize per-window memory for all satellites
        h_sat_memory = torch.zeros(self.num_total_sats, self.hidden_dim, device=device)
        c_sat_memory = torch.zeros(self.num_total_sats, self.hidden_dim, device=device)

        h_rec = torch.zeros(1, self.hidden_dim, device=device)
        c_rec = torch.zeros(1, self.hidden_dim, device=device)

        # Iterate through all snapshots in the window
        for snapshot in window_snapshots:
            snapshot = snapshot.to(device)
            x_dict = snapshot.x_dict
            eidx = snapshot.edge_index_dict

            # satellite_s_ids is stored under double-key in PyG NodeStorage
            s_ids_raw = snapshot['satellite_s_ids']['satellite_s_ids']
            if not isinstance(s_ids_raw, torch.Tensor):
                s_ids_raw = torch.tensor(s_ids_raw, dtype=torch.long)
            s_ids = s_ids_raw.to(device).reshape(-1)

            # READ: select only currently visible satellites from memory bank
            if s_ids.numel() > 0:
                h_sat_active = torch.index_select(h_sat_memory, 0, s_ids)
                c_sat_active = torch.index_select(c_sat_memory, 0, s_ids)
            else:
                # No satellites visible — active state is empty 
                h_sat_active = torch.empty((0, self.hidden_dim), device=device)
                c_sat_active = torch.empty((0, self.hidden_dim), device=device)

            h_dict_in = {'receiver': h_rec, 'satellite': h_sat_active}
            c_dict_in = {'receiver': c_rec, 'satellite': c_sat_active}

            # LSTM STEP
            h_out, c_out = self.gclstm(x_dict, eidx, h_dict_in, c_dict_in)

            # Update receiver state
            h_rec = h_out['receiver']
            c_rec = c_out['receiver']

            # updated states back to memory bank
            if s_ids.numel() > 0:
                h_sat_memory = h_sat_memory.index_put((s_ids,), h_out['satellite'])
                c_sat_memory = c_sat_memory.index_put((s_ids,), c_out['satellite'])

        # Predict from final receiver hidden state
        pred = self.linear_out(self.dropout(h_rec))
        true = window_snapshots[-1].y_dict['receiver']

        return pred, true