"""
Model evaluation, inference, and visualization utilities.
Shared by train.py and sustainability.py.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data import load_data
from lstm_model import LSTMModel
import config


def load_trained_model(device=None):
    """Load best_model.pth and reconstruct model + data."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    _, test_loader, dataset = load_data()

    checkpoint = torch.load('best_model.pth', map_location=device,
                            weights_only=False)

    input_size = len(dataset.input_features)
    thermal_output_size = dataset.thermal_features.shape[1]

    model = LSTMModel(
        input_size=input_size,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        thermal_output_size=thermal_output_size
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, test_loader, dataset, device


def compute_predictions(model, test_loader, dataset, device):
    """Forward-pass BTP predictions on test set (physical units)."""
    model.eval()
    y_scale = dataset.y_scaler.scale_[0]
    y_mean = dataset.y_scaler.mean_[0]

    btp_preds, btp_trues = [], []

    for batch in test_loader:
        X_batch = batch['X'].to(device).unsqueeze(1)
        with torch.no_grad():
            btp_pred, _ = model(X_batch)
            btp_pred_phys = btp_pred.squeeze(1).cpu().numpy() * y_scale + y_mean
            y_true = batch['Y'].cpu().numpy()
            btp_trues.extend((y_true * y_scale + y_mean).tolist())
            btp_preds.extend(btp_pred_phys.tolist())

    return np.array(btp_trues), np.array(btp_preds)


def compute_model_sensitivity(model, test_loader, dataset, device):
    """Per-sample d(BTP)/d(fuel) via backprop (physical units)."""
    model.eval()
    input_cols = list(dataset.input_features)

    coke_indices = [i for i, col in enumerate(input_cols) if col == '焦粉']
    coal_indices = [i for i, col in enumerate(input_cols)
                    if col == '烧结用白煤']
    fuel_indices = coke_indices + coal_indices
    print(f"  Fuel feature indices: {fuel_indices}")

    X_scale = dataset.scaler_combined.scale_
    y_scale = dataset.y_scaler.scale_[0]

    def to_physical(dy_dx_scaled, x_idx):
        return dy_dx_scaled * (y_scale / X_scale[x_idx])

    all_sensitivities = []

    for batch in test_loader:
        X_batch = batch['X'].to(device).unsqueeze(1)
        X_batch.requires_grad_(True)

        btp_pred, _ = model(X_batch)

        grad_all = torch.autograd.grad(
            btp_pred.sum(), X_batch, retain_graph=False
        )[0]

        fuel_grads_scaled = grad_all[:, 0, fuel_indices].sum(dim=1)
        fuel_grads_scaled = fuel_grads_scaled.detach().cpu().numpy()

        sens_phys = np.array([to_physical(g, fuel_indices[0])
                              for g in fuel_grads_scaled])
        all_sensitivities.extend(sens_phys.tolist())

        X_batch.requires_grad_(False)

    sensitivities = np.array(all_sensitivities)
    print(f"  Model sensitivity: mean={np.mean(sensitivities):.6f}, "
          f"std={np.std(sensitivities):.6f} m/(kg/t)")
    return sensitivities


def evaluate_model(model, data_loader, dataset, device):
    """Evaluate model and compute metrics in physical units."""
    model.eval()
    all_btp_pred, all_btp_true = [], []

    with torch.no_grad():
        for batch in data_loader:
            X_batch = batch['X'].to(device).unsqueeze(1)
            Y_batch = batch['Y'].to(device)

            btp_pred, _ = model(X_batch)

            btp_pred_orig = dataset.y_scaler.inverse_transform(
                btp_pred.cpu().numpy()
            )
            btp_true_orig = dataset.y_scaler.inverse_transform(
                Y_batch.cpu().numpy().reshape(-1, 1)
            )

            all_btp_pred.extend(btp_pred_orig.flatten())
            all_btp_true.extend(btp_true_orig.flatten())

    btp_true = np.array(all_btp_true)
    btp_pred = np.array(all_btp_pred)
    mse = mean_squared_error(btp_true, btp_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(btp_true, btp_pred)
    r2 = r2_score(btp_true, btp_pred)

    print(f"\n  Test set metrics:")
    print(f"    BTP RMSE: {rmse:.4f}")
    print(f"    BTP MAE:  {mae:.4f}")
    print(f"    BTP R²:   {r2:.4f}")

    return all_btp_pred, btp_true, rmse, r2


def visualize_results(all_labels, all_preds):
    """Generate prediction scatter, error distribution, and line plots."""
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False

    labels_np = np.array(all_labels)
    preds_np = np.array(all_preds)
    errors = preds_np - labels_np

    # ── Scatter: predicted vs actual ────────────────────────────────
    plt.figure(figsize=(8, 6))
    plt.scatter(all_labels, all_preds, alpha=0.6, color='steelblue',
                label='Predicted')
    mmin = min(labels_np.min(), preds_np.min()) - 0.1
    mmax = max(labels_np.max(), preds_np.max()) + 0.1
    plt.plot([mmin, mmax], [mmin, mmax], 'r--', linewidth=2, label='y=x')
    plt.xlabel('Actual BTP', fontsize=12)
    plt.ylabel('Predicted BTP', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('btp_prediction_scatterLSTM-PI.png', dpi=1200)
    plt.show()

    # ── Error distribution ──────────────────────────────────────────
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=30, color='lightseagreen', alpha=0.7,
             edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2,
                label='Zero Error')
    plt.xlabel('Prediction Error', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("btp_error_distribution.png", dpi=1200, bbox_inches='tight')
    plt.show()

    # ── Line plot: actual vs predicted ──────────────────────────────
    x_idx = np.arange(len(all_labels))
    plt.figure(figsize=(12, 6))
    plt.plot(x_idx, labels_np, color='steelblue', linewidth=2,
             label='Actual BTP', marker='o', markersize=4)
    plt.plot(x_idx, preds_np, color='orangered', linewidth=2,
             label='Predicted BTP', marker='s', markersize=4, alpha=0.8)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.xlabel('Sample', fontsize=12)
    plt.ylabel('BTP Value', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3, linestyle='--')
    plt.savefig("btp_prediction_line.png", dpi=1200, bbox_inches='tight')
    plt.show()
