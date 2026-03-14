"""
Data Generator for Project 007: Anomaly Detection & Dimensionality Reduction
  A) Jet Engine Sensor Anomalies (general, intuitive)
  B) Silicon Thermal Hotspot Detection (Posiva)
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def generate_jet_engine_data(n_normal=4500, n_anomaly=150, seed=42):
    """
    Jet Engine Sensor Anomaly Detection
    Normal: 4500 flights, Anomaly: 150 (3.2%)
    Features: turbine_temp_c, oil_pressure_psi, vibration_hz,
              fuel_flow_rate, exhaust_gas_temp_c, fan_speed_rpm,
              compressor_ratio, n1_speed_pct
    """
    rng = np.random.default_rng(seed)

    # Normal operation
    normal = pd.DataFrame({
        'turbine_temp_c':    rng.normal(850, 30, n_normal),
        'oil_pressure_psi':  rng.normal(55, 5, n_normal),
        'vibration_hz':      rng.normal(120, 10, n_normal),
        'fuel_flow_rate':    rng.normal(2200, 150, n_normal),
        'exhaust_gas_temp_c': rng.normal(620, 25, n_normal),
        'fan_speed_rpm':     rng.normal(3200, 80, n_normal),
        'compressor_ratio':  rng.normal(28.5, 1.2, n_normal),
        'n1_speed_pct':      rng.normal(88, 3, n_normal),
        'label': 0  # Normal
    })

    # Anomalies: 3 types
    n_each = n_anomaly // 3

    # Type 1: Turbine overheat
    overheat = pd.DataFrame({
        'turbine_temp_c':    rng.normal(1050, 40, n_each),
        'oil_pressure_psi':  rng.normal(48, 8, n_each),
        'vibration_hz':      rng.normal(130, 15, n_each),
        'fuel_flow_rate':    rng.normal(2600, 200, n_each),
        'exhaust_gas_temp_c': rng.normal(780, 40, n_each),
        'fan_speed_rpm':     rng.normal(3400, 100, n_each),
        'compressor_ratio':  rng.normal(30.5, 1.5, n_each),
        'n1_speed_pct':      rng.normal(95, 4, n_each),
        'label': 1
    })

    # Type 2: Oil pressure loss
    oil_loss = pd.DataFrame({
        'turbine_temp_c':    rng.normal(870, 35, n_each),
        'oil_pressure_psi':  rng.normal(28, 6, n_each),
        'vibration_hz':      rng.normal(145, 20, n_each),
        'fuel_flow_rate':    rng.normal(2250, 180, n_each),
        'exhaust_gas_temp_c': rng.normal(640, 30, n_each),
        'fan_speed_rpm':     rng.normal(3150, 90, n_each),
        'compressor_ratio':  rng.normal(27.8, 1.3, n_each),
        'n1_speed_pct':      rng.normal(86, 4, n_each),
        'label': 1
    })

    # Type 3: Fan blade imbalance (high vibration)
    vibration = pd.DataFrame({
        'turbine_temp_c':    rng.normal(860, 30, n_anomaly - 2*n_each),
        'oil_pressure_psi':  rng.normal(54, 5, n_anomaly - 2*n_each),
        'vibration_hz':      rng.normal(195, 25, n_anomaly - 2*n_each),
        'fuel_flow_rate':    rng.normal(2280, 160, n_anomaly - 2*n_each),
        'exhaust_gas_temp_c': rng.normal(630, 28, n_anomaly - 2*n_each),
        'fan_speed_rpm':     rng.normal(3350, 120, n_anomaly - 2*n_each),
        'compressor_ratio':  rng.normal(28.2, 1.4, n_anomaly - 2*n_each),
        'n1_speed_pct':      rng.normal(89, 3, n_anomaly - 2*n_each),
        'label': 1
    })

    df = pd.concat([normal, overheat, oil_loss, vibration], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df.round(2)
    df.to_csv(DATA_DIR / "jet_engine_sensors.csv", index=False)
    print(f"Jet Engine dataset: {len(df)} rows, {df['label'].mean()*100:.1f}% anomalies")
    return df


def generate_thermal_hotspot_data(n_normal=9000, n_anomaly=300, seed=77):
    """
    Silicon Thermal Hotspot Detection (Post-Silicon Validation)
    Features: core_temp_c, power_density_mw_mm2, thermal_resistance,
              leakage_current_ua, clock_freq_ghz, vdd_mv,
              ir_drop_mv, thermal_gradient
    """
    rng = np.random.default_rng(seed)

    # Normal operation
    normal = pd.DataFrame({
        'core_temp_c':          rng.normal(75, 8, n_normal),
        'power_density_mw_mm2': rng.normal(120, 15, n_normal),
        'thermal_resistance':   rng.normal(0.45, 0.05, n_normal),
        'leakage_current_ua':   rng.normal(850, 80, n_normal),
        'clock_freq_ghz':       rng.normal(3.2, 0.2, n_normal),
        'vdd_mv':               rng.normal(950, 25, n_normal),
        'ir_drop_mv':           rng.normal(18, 4, n_normal),
        'thermal_gradient':     rng.normal(12, 3, n_normal),
        'label': 0
    })

    # Hotspot anomalies: 3 types
    n_each = n_anomaly // 3

    # Type 1: Thermal runaway
    runaway = pd.DataFrame({
        'core_temp_c':          rng.normal(115, 12, n_each),
        'power_density_mw_mm2': rng.normal(210, 25, n_each),
        'thermal_resistance':   rng.normal(0.72, 0.08, n_each),
        'leakage_current_ua':   rng.normal(2200, 200, n_each),
        'clock_freq_ghz':       rng.normal(2.8, 0.3, n_each),
        'vdd_mv':               rng.normal(1020, 30, n_each),
        'ir_drop_mv':           rng.normal(45, 8, n_each),
        'thermal_gradient':     rng.normal(38, 6, n_each),
        'label': 1
    })

    # Type 2: IR drop hotspot
    ir_hotspot = pd.DataFrame({
        'core_temp_c':          rng.normal(88, 10, n_each),
        'power_density_mw_mm2': rng.normal(155, 18, n_each),
        'thermal_resistance':   rng.normal(0.52, 0.06, n_each),
        'leakage_current_ua':   rng.normal(1100, 100, n_each),
        'clock_freq_ghz':       rng.normal(3.0, 0.25, n_each),
        'vdd_mv':               rng.normal(880, 35, n_each),
        'ir_drop_mv':           rng.normal(72, 12, n_each),
        'thermal_gradient':     rng.normal(22, 5, n_each),
        'label': 1
    })

    # Type 3: Leakage spike
    leakage = pd.DataFrame({
        'core_temp_c':          rng.normal(95, 10, n_anomaly - 2*n_each),
        'power_density_mw_mm2': rng.normal(175, 20, n_anomaly - 2*n_each),
        'thermal_resistance':   rng.normal(0.58, 0.07, n_anomaly - 2*n_each),
        'leakage_current_ua':   rng.normal(3500, 400, n_anomaly - 2*n_each),
        'clock_freq_ghz':       rng.normal(3.1, 0.22, n_anomaly - 2*n_each),
        'vdd_mv':               rng.normal(970, 28, n_anomaly - 2*n_each),
        'ir_drop_mv':           rng.normal(28, 6, n_anomaly - 2*n_each),
        'thermal_gradient':     rng.normal(28, 5, n_anomaly - 2*n_each),
        'label': 1
    })

    df = pd.concat([normal, runaway, ir_hotspot, leakage], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df.round(3)
    df.to_csv(DATA_DIR / "silicon_thermal_hotspots.csv", index=False)
    print(f"Silicon Thermal dataset: {len(df)} rows, {df['label'].mean()*100:.1f}% anomalies")
    return df


if __name__ == "__main__":
    generate_jet_engine_data()
    generate_thermal_hotspot_data()
    print("Both datasets generated successfully.")
