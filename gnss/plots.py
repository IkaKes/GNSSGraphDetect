import matplotlib.pyplot as plt

def plot_loss_curves(epochs, train_losses, test_losses, save_path=None):
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, test_losses,  marker='o', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('SmoothL1 Loss')
    plt.title('Training vs. Validation Loss by Epoch')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_mae_multi_curve(epochs, mae_lat, mae_lon, mae_sum, save_path=None):
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, mae_lat, marker='o', color='tab:blue', label='Val MAE Lat (cm)')
    plt.plot(epochs, mae_lon, marker='o', color='tab:green', label='Val MAE Lon (cm)')
    plt.plot(epochs, mae_sum, marker='o', color='tab:orange', label='Val MAE Sum (cm)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (cm)')
    plt.title('Validation MAE by Epoch (Lat, Lon, Sum)')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()