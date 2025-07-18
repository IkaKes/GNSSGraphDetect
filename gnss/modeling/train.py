import torch
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np
from dvclive.lightning import DVCLiveLogger
from gnss.dataset import build_loader
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import os
import pandas as pd

from gnss.model import FullModel
from gnss.model2 import FullModelTwoLayer


class LightningFullModel(pl.LightningModule):

    def __init__(self, model_class, in_channels_dict, hidden_dim, metadata, 
                 scalers_tv, scalers_test, 
                 initial_lr, weight_decay_val):
        super().__init__()
        self.save_hyperparameters('hidden_dim', 'initial_lr', 'weight_decay_val')
        self.model = model_class(in_channels_dict, hidden_dim, metadata)
        self.loss_fn = nn.SmoothL1Loss(beta=1e-2)
        
        self.scalers_for_train_val = scalers_tv
        self.scalers_for_test = scalers_test

    def forward(self, window_snapshots):
        return self.model(window_snapshots, device=self.device)

    def _calculate_loss(self, pred_norm, true_norm):
        lat_p, lon_p = pred_norm[:, 0], pred_norm[:, 1]
        lat_t, lon_t = true_norm[:, 0], true_norm[:, 1]
        loss_lat = self.loss_fn(lat_p.unsqueeze(1), lat_t.unsqueeze(1))
        loss_lon = self.loss_fn(lon_p.unsqueeze(1), lon_t.unsqueeze(1))
        return loss_lat + loss_lon
    
    def training_step(self, batch, batch_idx):
        pred_norm, true_norm = self(batch)
        loss = self._calculate_loss(pred_norm, true_norm)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        return loss
        
    def validation_step(self, batch, batch_idx):
        pred_norm, true_norm = self(batch)
        loss = self._calculate_loss(pred_norm, true_norm)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, batch_size=1)

        targ_scaler = self.scalers_for_train_val['targ']
        pred_cm = targ_scaler.inverse_transform(pred_norm.cpu().numpy())
        true_cm = targ_scaler.inverse_transform(true_norm.cpu().numpy())
        
        mae_lat = np.abs(pred_cm[:, 0] - true_cm[:, 0]).mean()
        mae_lon = np.abs(pred_cm[:, 1] - true_cm[:, 1]).mean()
        diffs = np.sqrt((pred_cm[:, 0] - true_cm[:, 0])**2 + (pred_cm[:, 1] - true_cm[:, 1])**2)
        mae_sum = diffs.mean()

        self.log('val_mae_lat_cm', mae_lat, on_epoch=True, prog_bar=False, batch_size=1)
        self.log('val_mae_lon_cm', mae_lon, on_epoch=True, prog_bar=False, batch_size=1)
        self.log('val_mae_sum_cm', mae_sum, on_epoch=True, prog_bar=True, batch_size=1)
        return loss
        
    def test_step(self, batch, batch_idx):
        pred_norm, true_norm = self(batch)
        loss = self._calculate_loss(pred_norm, true_norm)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, batch_size=1)

        targ_scaler = self.scalers_for_test['targ']
        pred_cm = targ_scaler.inverse_transform(pred_norm.cpu().numpy())
        true_cm = targ_scaler.inverse_transform(true_norm.cpu().numpy())
        
        mae_lat = np.abs(pred_cm[:, 0] - true_cm[:, 0]).mean()
        mae_lon = np.abs(pred_cm[:, 1] - true_cm[:, 1]).mean()
        diffs = np.sqrt((pred_cm[:, 0] - true_cm[:, 0])**2 + (pred_cm[:, 1] - true_cm[:, 1])**2)
        mae_sum = diffs.mean()

        self.log('test_mae_lat_cm', mae_lat, on_epoch=True, prog_bar=False, batch_size=1)
        self.log('test_mae_lon_cm', mae_lon, on_epoch=True, prog_bar=False, batch_size=1)
        self.log('test_mae_sum_cm', mae_sum, on_epoch=True, prog_bar=True, batch_size=1)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.initial_lr, weight_decay=self.hparams.weight_decay_val)
        return optimizer



def train_lightning_model(
    train_signals, val_signals, test_signals, 
    scalers_for_train_val, scalers_for_test, 
    metadata, config
):
    prepare_cfg = config['prepare_data']
    train_cfg = config['train']
    
    window_size = prepare_cfg['window_size']
    stride = prepare_cfg['stride']

    train_loader = build_loader(train_signals, window_size, shuffle=True, stride=stride)
    val_loader = build_loader(val_signals, window_size, shuffle=False, stride=stride)
    test_loader = build_loader(test_signals, window_size, shuffle=False, stride=stride)
    
    if train_cfg['model_name'] == 'FullModel':
        model_class_to_use = FullModel
    else:
        model_class_to_use = FullModelTwoLayer
    
    model = LightningFullModel(
        model_class=model_class_to_use, 
        in_channels_dict={'receiver': 2, 'satellite': 3},
        hidden_dim=train_cfg['hidden_dim'], 
        metadata=metadata, 
        scalers_tv=scalers_for_train_val,  
        scalers_test=scalers_for_test,
        initial_lr=train_cfg['initial_lr'], 
        weight_decay_val=train_cfg['weight_decay_val']
    )
    
    early_stop_callback = EarlyStopping(**train_cfg['early_stopping'])
    

    checkpoint_callback = ModelCheckpoint(
        dirpath=train_cfg['output_dir'], # Ova putanja dolazi iz 'train' sekcije
        filename='best_model',
        save_top_k=1,
        monitor=train_cfg['early_stopping']['monitor'],
        mode=train_cfg['early_stopping']['mode']
    )    


    trainer = pl.Trainer(
        max_epochs=train_cfg['n_epochs'],
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=config['logger'],
        accelerator=train_cfg.get('accelerator'),
        devices=train_cfg.get('devices')
    )
    
    trainer.fit(model, train_loader, val_loader)

    print("\n--- Training done ---")
    print(f"Best model: {checkpoint_callback.best_model_path}")
    if checkpoint_callback.best_model_score:
        print(f"Best value/metric '{train_cfg['early_stopping']['monitor']}': {checkpoint_callback.best_model_score:.4f}")

    print("\n--- Testing on test set ---")
    trainer.test(dataloaders=test_loader, ckpt_path='best', verbose=True)

    final_metrics = trainer.callback_metrics
    print("\n--- Final metrics on the test set ---")
    print(final_metrics)
    
    return trainer, checkpoint_callback.best_model_path, final_metrics