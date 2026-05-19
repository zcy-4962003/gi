"""
Physics-informed loss functions for sintering thermal balance.

Gas-phase and solid-phase PDE residual losses enforce physical
consistency of predicted thermal features.
"""
import torch
from torch import nn
import pandas as pd
import config


def extract_thermal_params(thermal_scaled_batch, dataset):
    """Extract physical parameters from predicted thermal features."""
    thermal_original = dataset.scaler_thermal.inverse_transform(
        thermal_scaled_batch.detach().cpu().numpy()
    )
    thermal_cols = dataset.thermal_features.columns.tolist()
    thermal_df = pd.DataFrame(thermal_original, columns=thermal_cols)

    batch_size = thermal_original.shape[0]
    thermal_params = []

    for b in range(batch_size):
        row = thermal_df.iloc[b]
        params = {}

        # Gas mass flow
        if '二号风机风量' in thermal_cols:
            fan_flow_h = row['二号风机风量']
            fan_flow_s = fan_flow_h / 3600
            params['f_g'] = config.RHO_AIR * fan_flow_s
        else:
            params['f_g'] = config.DEFAULT_F_G

        # Solid reaction heat from fuel
        fuel_sum = 0
        if '焦粉' in thermal_cols:
            fuel_sum += row['焦粉']
        if '烧结用白煤' in thermal_cols:
            fuel_sum += row['烧结用白煤']
        params['Q_s'] = (fuel_sum / 100) * config.Q_COAL if fuel_sum > 0 else config.DEFAULT_Q_S

        # Bed density
        params['f_s'] = config.DEFAULT_F_S

        # Gas reaction heat
        params['Q_g'] = 0

        # Gas temperature
        t_g_cols = [col for col in thermal_cols if '风箱废气温度' in col]
        if t_g_cols:
            params['T_g_data'] = row[t_g_cols].mean()
        else:
            params['T_g_data'] = None

        thermal_params.append(params)

    return thermal_params


def gas_thermal_loss(thermal_pred, thermal_params):
    """Gas-phase thermal balance residual loss."""
    batch_size = len(thermal_params)
    losses = []
    thermal_pred_detached = thermal_pred.detach()

    for b in range(batch_size):
        T_g16 = thermal_pred_detached[b, 0] if thermal_pred.shape[1] > 0 else 0
        T_g22 = thermal_pred_detached[b, 1] if thermal_pred.shape[1] > 1 else 0
        T_s_curr = thermal_pred_detached[b, 2] if thermal_pred.shape[1] > 2 else 0

        dT_g_dz = (T_g22 - T_g16) / config.DELTA_Z
        left = thermal_params[b]['f_g'] * config.C_PG * dT_g_dz
        T_g_avg = (T_g16 + T_g22) / 2
        right = -config.H_GS * config.A_S * (T_g_avg - T_s_curr) + thermal_params[b]['Q_g']

        losses.append(torch.abs(left - right) / 1e6)

    return torch.mean(torch.stack(losses))


def solid_thermal_loss(thermal_pred, thermal_params):
    """Solid-phase thermal balance residual loss."""
    batch_size = len(thermal_params)
    losses = []
    thermal_pred_detached = thermal_pred.detach()

    for b in range(batch_size):
        T_s_curr = thermal_pred_detached[b, 2] if thermal_pred.shape[1] > 2 else 0
        T_s_prev = thermal_pred_detached[b, 3] if thermal_pred.shape[1] > 3 else 0
        T_g16 = thermal_pred_detached[b, 0] if thermal_pred.shape[1] > 0 else 0
        T_g22 = thermal_pred_detached[b, 1] if thermal_pred.shape[1] > 1 else 0

        dT_s_dt = (T_s_curr - T_s_prev) / config.DELTA_T
        left = thermal_params[b]['f_s'] * config.C_PS * dT_s_dt
        T_g_avg = (T_g16 + T_g22) / 2
        right = -config.H_GS * config.A_S * (T_s_curr - T_g_avg) + thermal_params[b]['Q_s']

        losses.append(torch.abs(left - right) / 1e6)

    return torch.mean(torch.stack(losses))


def total_loss_fn(btp_pred, btp_true, thermal_pred, thermal_true, thermal_params):
    """Total loss = data_weight * L_data + physics_weight * L_physics."""
    btp_loss = nn.MSELoss()(btp_pred.squeeze(), btp_true)
    thermal_loss = nn.MSELoss()(thermal_pred, thermal_true)
    data_loss = btp_loss + thermal_loss

    gas_loss = gas_thermal_loss(thermal_pred, thermal_params)
    solid_loss = solid_thermal_loss(thermal_pred, thermal_params)
    physics_loss = gas_loss + solid_loss

    total = config.WEIGHT_DATA * data_loss + config.WEIGHT_PHYSICS * physics_loss
    return total, btp_loss, thermal_loss, physics_loss
