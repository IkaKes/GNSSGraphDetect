import numpy as np
import pandas as pd
import yaml
import os
import random
import torch
from pathlib import Path
import pickle
from gnss.modeling.train import train_lightning_model
from dvclive.lightning import DVCLiveLogger
from gnss.config import PARSED_DATA_DIR
#torch.set_float32_matmul_precision('high') 


def main():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    seed = params['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_dir = Path(params['train']['output_dir'])
    data_out_dir = Path(params['output_dir'])

    print(f"Loading data from: {data_out_dir}")
    train_signals = torch.load(data_out_dir / 'train_graphs.pt')
    val_signals   = torch.load(data_out_dir / 'val_graphs.pt')
    test_signals  = torch.load(data_out_dir / 'test_graphs.pt')
    
    print("Učitavanje dva seta scalera...")
    scalers_for_train_val = {
        'rec': torch.load(data_out_dir / 'rec_scaler_train.pt'),
        'sat': torch.load(data_out_dir / 'sat_scaler_train.pt'),
        'targ': torch.load(data_out_dir / 'targ_scaler_train.pt')
    }
    scalers_for_test = {
        'rec': torch.load(data_out_dir / 'rec_scaler_test.pt'),
        'sat': torch.load(data_out_dir / 'sat_scaler_test.pt'),
        'targ': torch.load(data_out_dir / 'targ_scaler_test.pt')
    }

    metadata = train_signals[0][0].metadata() if train_signals and train_signals[0] else None
    
    logger = DVCLiveLogger(
        dir=str(output_dir / "dvclive"),
        save_dvc_exp=False,
        dvcyaml=False
    )
    

    config = {
        **params, 
        'output_dir': str(output_dir),
        'logger': logger
    }

    trainer, best_model_path, best_metrics = train_lightning_model(
        train_signals, 
        val_signals, 
        test_signals,
        scalers_for_train_val,
        scalers_for_test,      
        metadata,
        config
    )

    if hasattr(logger, "finalize"):
        logger.finalize("success")

    if best_metrics:
        print("\n--- Final metrics on test set ---")
        metrics_to_save = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in best_metrics.items()}
        print(yaml.dump(metrics_to_save, indent=2))
        
        with open(output_dir / 'metrics.yaml', 'w') as f:
            yaml.dump(metrics_to_save, f)
    
    if best_model_path:
        print(f"\nBest model saved on: {best_model_path}")
    
    print(f"\nLogs saved on: {output_dir}")

if __name__ == '__main__':
    main()
