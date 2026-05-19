"""
Shared constants for RevIn-LSTM-PI project.
"""
# ── Physics Constants ──────────────────────────────────────────────────
RHO_AIR = 1.225        # kg/m^3
C_PG = 1005            # J/(kg·K)  gas specific heat
C_PS = 840             # J/(kg·K)  solid specific heat
H_GS = 150             # W/(m^2·K) gas-solid heat transfer coefficient
A_S = 300              # m^2/m^3  specific surface area
Q_COAL = 29308000      # J/kg  coal heating value
DELTA_Z = 0.5          # m  bed height increment
DELTA_T = 60           # s  time step
DEFAULT_F_S = 1800     # kg/m^3  default bed density
DEFAULT_F_G = 0.5      # kg/s  default gas mass flow
DEFAULT_Q_S = 1e6      # W/m^3  default solid heat source

# ── Training ───────────────────────────────────────────────────────────
WEIGHT_DATA = 0.64
WEIGHT_PHYSICS = 0.36
MAX_EPOCHS = 500
PATIENCE = 10
HIDDEN_SIZE = 64
NUM_LAYERS = 4
LEARNING_RATE = 2e-5

# ── Plant Assumptions (typical 360 m^2 sinter strand) ──────────────────
BTP_TARGET_M = 20.60
ANNUAL_PRODUCTION_T = 5_000_000     # 5 Mtpa sinter
CO2_EMISSION_FACTOR = 2.8           # kg CO2 per kg coal
COKE_PRICE_CNY = 1500               # CNY per tonne coke breeze
LITERATURE_FUEL_SENSITIVITY = 1.0   # kg coke / (t-sinter * m BTP)
