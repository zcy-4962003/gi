"""
Sustainability Scenario Comparison
===================================
Generates sustainability_scenario_comparison.png and
sustainability_scenario_scatter.png — Baseline vs. RevIn-LSTM-PI.
"""
import numpy as np
import matplotlib.pyplot as plt
import config
from evaluate import (load_trained_model, compute_predictions,
                      compute_model_sensitivity)


def compare_scenarios(btp_true, btp_pred):
    """Compute fuel savings from improved BTP prediction."""
    sigma_actual = np.std(btp_true)
    sigma_model = np.std(btp_true - btp_pred)
    rmse_model = np.sqrt(np.mean((btp_true - btp_pred) ** 2))
    mean_btp = np.mean(btp_true)

    fuel_sens = config.LITERATURE_FUEL_SENSITIVITY
    mean_offset = max(0, mean_btp - config.BTP_TARGET_M)
    savings_from_mean = mean_offset * fuel_sens

    practical_factor = 0.60
    current_variance_fuel = sigma_actual * fuel_sens
    achievable_sigma = max(sigma_model, sigma_actual * 0.05)
    savings_from_variance = (sigma_actual - achievable_sigma) * fuel_sens * practical_factor

    total_savings = savings_from_mean + savings_from_variance

    annual_fuel_saved_t = total_savings * config.ANNUAL_PRODUCTION_T / 1000
    annual_co2_reduced_t = annual_fuel_saved_t * config.CO2_EMISSION_FACTOR
    annual_cost_saved_10k = annual_fuel_saved_t * config.COKE_PRICE_CNY / 10000

    return {
        'btp_true': btp_true,
        'sigma_actual': sigma_actual,
        'sigma_model': sigma_model,
        'rmse_model': rmse_model,
        'current_variance_fuel_kg_per_t': current_variance_fuel,
        'savings_from_mean_kg_per_t': savings_from_mean,
        'savings_from_variance_kg_per_t': savings_from_variance,
        'total_fuel_savings_kg_per_t': total_savings,
        'annual_fuel_saved_tonnes': annual_fuel_saved_t,
        'annual_co2_reduced_tonnes': annual_co2_reduced_t,
        'annual_cost_saved_10k_cny': annual_cost_saved_10k,
    }


def generate_scenario_comparison(sc):
    """Bar chart: Baseline vs Model-Guided fuel consumption."""
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(9, 6.5))

    baseline = sc['current_variance_fuel_kg_per_t']
    savings = sc['total_fuel_savings_kg_per_t']
    savings = min(savings, baseline * 0.95)
    model_fuel = max(baseline * 0.05, baseline - savings)

    print(f"  baseline={baseline:.6f}  savings={savings:.6f}  "
          f"model_fuel={model_fuel:.6f}")

    labels = ['Baseline\n(No Model)', 'With RevIn-LSTM-PI\n(Model-Guided)']
    values = [baseline, model_fuel]
    colors = ['#E74C3C', '#27AE60']

    bars = ax.bar(labels, values, color=colors, width=0.45,
                  edgecolor='black', linewidth=1.2)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f'{val:.4f} kg/t', ha='center', va='bottom',
                fontsize=14, fontweight='bold')

    ax.set_ylabel('Fuel Consumption Rate (kg/t-sinter)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, baseline * 1.25)

    plt.tight_layout()
    plt.savefig('sustainability_scenario_comparison.png', dpi=600)
    plt.close()
    print("Saved: sustainability_scenario_comparison.png")


def generate_scenario_scatter(sc, model_sensitivity):
    """Scatter: Excess Fuel vs Deviation with per-sample model sensitivity."""
    btp_true = sc['btp_true']
    deviation = btp_true - config.BTP_TARGET_M

    over = deviation > 0
    excess_fuel = np.zeros_like(deviation)
    valid = over & (np.abs(model_sensitivity) > 1e-9)
    excess_fuel[valid] = deviation[valid] / model_sensitivity[valid]

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(10, 6.5))

    scat = ax.scatter(deviation, excess_fuel,
                      c=excess_fuel, cmap='YlOrRd', alpha=0.6,
                      edgecolors='black', linewidth=0.15, s=25)

    ax.axvline(x=0, color='#2C3E50', linestyle='--', linewidth=1.2)
    ax.axhline(y=0, color='#2C3E50', linestyle='--', linewidth=1.2)

    cbar = plt.colorbar(scat, ax=ax, pad=0.02)
    cbar.set_label('Excess Fuel Intensity (kg/t-sinter)', fontsize=10)

    ax.set_xlabel('BTP Deviation from Target (m)', fontsize=12)
    ax.set_ylabel('Excess Fuel Consumption (kg/t-sinter)', fontsize=12)
    ax.set_title(f'Excess Fuel vs. BTP Deviation (Model-Driven Sensitivity)\n'
                 f'(RMSE = {sc["rmse_model"]:.4f} m, '
                 f'σ_error = {sc["sigma_model"]:.4f} m)',
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('sustainability_scenario_scatter.png', dpi=600)
    plt.close()
    print("Saved: sustainability_scenario_scatter.png")


def main():
    print("=" * 55)
    print("SUSTAINABILITY SCENARIO COMPARISON")
    print("=" * 55)

    print("\n[1/3] Loading model and data...")
    model, test_loader, dataset, device = load_trained_model()

    print("\n[2/3] Computing predictions and model sensitivity...")
    btp_true, btp_pred = compute_predictions(model, test_loader, dataset, device)
    model_sens = compute_model_sensitivity(model, test_loader, dataset, device)

    print("\n[3/3] Generating figures...")
    sc = compare_scenarios(btp_true, btp_pred)
    generate_scenario_comparison(sc)
    generate_scenario_scatter(sc, model_sens)

    print(f"\n  Fuel savings: {sc['total_fuel_savings_kg_per_t']:.4f} kg/t")
    print(f"  Annual CO₂:   {sc['annual_co2_reduced_tonnes']:,.0f} tonnes")
    print("Done.")


if __name__ == "__main__":
    main()
