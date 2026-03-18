import yaml
import os
import random
import torch
import numpy as np
#import time
from pathlib import Path

from gnss.train.train import train_lightning_model
from dvclive.lightning import DVCLiveLogger

torch.set_float32_matmul_precision('high')


def main():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    seed = params['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_dir   = Path(params['train']['output_dir'])
    data_out_dir = Path(params['output_dir'])

    train_signals = torch.load(data_out_dir / 'train_graphs.pt', weights_only=False)
    val_signals   = torch.load(data_out_dir / 'val_graphs.pt',   weights_only=False)
    test_signals  = torch.load(data_out_dir / 'test_graphs.pt',  weights_only=False)

    scalers_for_train_val = {
        'rec':  torch.load(data_out_dir / 'rec_scaler_train.pt',  weights_only=False),
        'sat':  torch.load(data_out_dir / 'sat_scaler_train.pt',  weights_only=False),
        'targ': torch.load(data_out_dir / 'targ_scaler_train.pt', weights_only=False)
    }
    scalers_for_test = {
        'rec':  torch.load(data_out_dir / 'rec_scaler_test.pt',  weights_only=False),
        'sat':  torch.load(data_out_dir / 'sat_scaler_test.pt',  weights_only=False),
        'targ': torch.load(data_out_dir / 'targ_scaler_test.pt', weights_only=False)
    }

    num_total_sats = torch.load(data_out_dir / 'num_total_sats.pt', weights_only=False)

    # Extract graph metadata from first training snapshot
    metadata = train_signals[0][0].metadata() if train_signals and train_signals[0] else None
    if not metadata:
        raise ValueError("Could not extract metadata from training signals.")

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

    
    print("\nINFO - Starting training...")
    start_time = time.time()

    trainer, best_model_path, best_metrics = train_lightning_model(
        train_signals, val_signals, test_signals,
        scalers_for_train_val, scalers_for_test,
        metadata, config, num_total_sats=num_total_sats
    )


    if hasattr(logger, "finalize"):
        logger.finalize("success")

    if best_metrics:
        print("\n--- Final metrics on test set ---")
        metrics_to_save = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in best_metrics.items()
        }
        print(yaml.dump(metrics_to_save, indent=2))
        with open(output_dir / 'metrics.yaml', 'w') as f:
            yaml.dump(metrics_to_save, f)

    if best_model_path:
        print(f"\nBest model saved at: {best_model_path}")

    print(f"\nLogs saved to: {output_dir}")


if __name__ == '__main__':
    main()