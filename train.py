"""
RevIn-LSTM-PI Training Pipeline
================================
- RevIN handles input distribution shift
- Physics-informed loss: gas/solid thermal balance constraints
- Adam + CosineAnnealingLR + Early Stopping
"""
import torch
import torch.optim as optim
from data import load_data
from lstm_model import LSTMModel
from physics import extract_thermal_params, total_loss_fn
from evaluate import evaluate_model, visualize_results
import config


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    train_loader, test_loader, dataset = load_data()

    input_size = len(dataset.input_features)
    thermal_output_size = dataset.thermal_features.shape[1]

    model = LSTMModel(
        input_size=input_size,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        thermal_output_size=thermal_output_size
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                           weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.MAX_EPOCHS, eta_min=1e-6
    )

    print(f"\n  Model config:")
    print(f"    Input size: {input_size}")
    print(f"    Hidden size: {config.HIDDEN_SIZE}")
    print(f"    Num layers: {config.NUM_LAYERS}")
    print(f"    Learning rate: {config.LEARNING_RATE}")
    print(f"    Loss weights: data={config.WEIGHT_DATA}, "
          f"physics={config.WEIGHT_PHYSICS}")
    print(f"\n  Training...")

    best_rmse = float('inf')
    no_improve = 0

    for epoch in range(config.MAX_EPOCHS):
        model.train()
        epoch_total, epoch_btp, epoch_thermal, epoch_physics = 0, 0, 0, 0

        for batch in train_loader:
            X_batch = batch['X'].to(device).unsqueeze(1)
            Y_batch = batch['Y'].to(device)
            thermal_true = batch['thermal_true'].to(device)

            btp_pred, thermal_pred = model(X_batch)

            thermal_params = extract_thermal_params(thermal_pred, dataset)

            total_loss, btp_loss, thermal_loss, physics_loss = total_loss_fn(
                btp_pred, Y_batch, thermal_pred, thermal_true, thermal_params
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_total += total_loss.item()
            epoch_btp += btp_loss.item()
            epoch_thermal += thermal_loss.item()
            epoch_physics += physics_loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"\n  Epoch [{epoch + 1}/{config.MAX_EPOCHS}]")
            print(f"    Total: {epoch_total:.4f}  BTP: {epoch_btp:.4f}  "
                  f"Thermal: {epoch_thermal:.4f}  Physics: {epoch_physics:.4f}")

            _, _, rmse, r2 = evaluate_model(
                model, test_loader, dataset, device
            )

            if rmse < best_rmse:
                best_rmse = rmse
                no_improve = 0
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_rmse': best_rmse,
                    'input_features': dataset.input_features,
                    'thermal_output_size': thermal_output_size,
                    'scaler_y_mean': dataset.y_scaler.mean_.tolist(),
                    'scaler_y_scale': dataset.y_scaler.scale_.tolist(),
                }, 'best_model.pth')
                print(f"    Saved best model (RMSE={best_rmse:.4f})")
            else:
                no_improve += 1
                print(f"    No improvement ({no_improve}/{config.PATIENCE})")
                if no_improve >= config.PATIENCE:
                    print("\n  Early stopping triggered.")
                    break

        scheduler.step()

    # ── Final evaluation ──────────────────────────────────────────────
    print("\n  Loading best model for final evaluation...")
    checkpoint = torch.load('best_model.pth', map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    all_preds, all_labels, rmse, r2 = evaluate_model(
        model, test_loader, dataset, device
    )
    visualize_results(all_labels, all_preds)


if __name__ == "__main__":
    main()
