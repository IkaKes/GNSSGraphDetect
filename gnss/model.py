import torch
import torch.nn as nn
import numpy as np
from torch_geometric_temporal.nn.hetero import HeteroGCLSTM


class FullModel(nn.Module):

    def __init__(self, in_channels_dict, hidden_dim, metadata, dropout_rate=0.1):
        super().__init__()
        
        self.gclstm = HeteroGCLSTM(
            in_channels_dict=in_channels_dict,
            out_channels=hidden_dim,
            metadata=metadata
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.linear_out = nn.Linear(hidden_dim, 2)

    def forward(self, window_snapshots, device):
     
        hidden_dim = self.linear_out.in_features
        

        h_state = {'receiver': torch.zeros(hidden_dim, device=device)}
        c_state = {'receiver': torch.zeros(hidden_dim, device=device)}

        for snapshot in window_snapshots[:-1]:
            x_dict_for_gclstm = snapshot.x_dict
            eidx_on_device = snapshot.edge_index_dict
            
            s_ids_val = snapshot['satellite_s_ids']['satellite_s_ids']
            

            num_sat = s_ids_val.shape[0]
            
            h_sat = torch.zeros((num_sat, hidden_dim), device=device)
            c_sat = torch.zeros((num_sat, hidden_dim), device=device)

            rec_h = h_state['receiver'].unsqueeze(0)
            rec_c = c_state['receiver'].unsqueeze(0)

            for j, sid_tensor in enumerate(s_ids_val):
                sid_key = sid_tensor.item()
                if sid_key in h_state:
                    h_sat[j] = h_state[sid_key]
                    c_sat[j] = c_state[sid_key]
            
            h_dict_step = {'receiver': rec_h, 'satellite': h_sat}
            c_dict_step = {'receiver': rec_c, 'satellite': c_sat}
            
            h_out, c_out = self.gclstm(x_dict_for_gclstm, eidx_on_device, h_dict_step, c_dict_step)
            
            h_state['receiver'] = h_out['receiver'][0]
            c_state['receiver'] = c_out['receiver'][0]
            
            for j, sid_tensor in enumerate(s_ids_val):
                sid_key = sid_tensor.item()
                h_state[sid_key] = h_out['satellite'][j]
                c_state[sid_key] = c_out['satellite'][j]
        
        h_final = h_state['receiver'].unsqueeze(0)
        h_dropped = self.dropout(h_final)
        pred_norm = self.linear_out(h_dropped)
        true_norm = window_snapshots[-1].y_dict['receiver']
        
        return pred_norm, true_norm

