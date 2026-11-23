"""
AquaSentinel AI - Hydraulic Simulation & Leak Analysis Test Suite
Performs end-to-end execution of baseline vs leak scenarios in WNTR,
exports time-series CSVs, and prints rigorous comparative analytics.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from simulation.network_builder import build_network, get_network_summary, get_nodes_by_zone
from simulation.hydraulic_engine import HydraulicEngine
from simulation.fault_injection import FaultInjector


def print_header(title: str):
    width = 78
    print("\n" + "=" * width)
    print(f" {title.upper()} ".center(width, "="))
    print("=" * width)


def print_section(title: str):
    width = 78
    print(f"\n--- {title} " + "-" * max(0, width - len(title) - 5))


def run_test():
    print_header("AquaSentinel AI - Hydraulic Simulation Pipeline")

    # -------------------------------------------------------------
    # Step 1: Build the Water Network
    # -------------------------------------------------------------
    print_section("1. Network Topology Construction")
    simulation_duration_hours = 24
    wn_normal = build_network(
        duration_hours=simulation_duration_hours,
        hydraulic_timestep_hours=1,
    )
    summary = get_network_summary(wn_normal)

    print(f"  Network Model Name:         AquaSentinel 3-Zone Synthetic DMA")
    print(f"  Reservoirs:                 {summary['num_reservoirs']} (R1 @ 60.0m Head)")
    print(f"  Junctions:                  {summary['num_junctions']} (Across 3 Pressure Zones)")
    print(f"  Pipes:                      {summary['num_pipes']}")
    print(f"  Supply Pumps:               {summary['num_pumps']} (PUMP_1 Lift, PUMP_2 Booster)")
    print(f"  Isolation Valves:           {summary['num_valves']} (TCV Isolation on A, B, C)")
    print(f"  Total Network Links:        {summary['num_links']}")
    print(f"  Total Base Consumer Demand: {summary['total_base_demand_lps']:.1f} L/s")
    print(f"  Zone Distribution:          {summary['zones']}")

    zone_a_nodes = get_nodes_by_zone(wn_normal, "Zone_A")
    zone_b_nodes = get_nodes_by_zone(wn_normal, "Zone_B")
    zone_c_nodes = get_nodes_by_zone(wn_normal, "Zone_C")
    print(f"    Zone A Nodes: {zone_a_nodes}")
    print(f"    Zone B Nodes: {zone_b_nodes}")
    print(f"    Zone C Nodes: {zone_c_nodes}")

    # -------------------------------------------------------------
    # Step 2: Baseline (Normal) Simulation
    # -------------------------------------------------------------
    print_section("2. Executing Baseline (Normal) Simulation")
    engine_normal = HydraulicEngine(wn_normal)
    results_normal = engine_normal.run(engine="WNTR", mode="PDD")
    print(f"  [OK] Normal Simulation Completed Successfully.")
    print(f"       Engine:              {results_normal.engine_type} (Pressure-Dependent Demand)")
    print(f"       Timesteps Computed:  {len(results_normal.pressure.index)} steps (0 to {simulation_duration_hours} hrs)")
    print(f"       Mean System Head:    {results_normal.head.mean().mean():.2f} m")
    print(f"       Mean Pressure:       {results_normal.pressure.mean().mean():.2f} m")

    # -------------------------------------------------------------
    # Step 3: Inject Leak into Zone B
    # -------------------------------------------------------------
    print_section("3. Fault Injection: Zone B Burst Leak")
    target_junction = "J_B2"
    leak_orifice_area = 0.0025  # m^2 (~56 mm diameter circular orifice)
    leak_start_time = 0         # active across simulation

    wn_leak = build_network(
        duration_hours=simulation_duration_hours,
        hydraulic_timestep_hours=1,
    )
    leak_meta = FaultInjector.inject_zone_b_leak(
        wn=wn_leak,
        junction_name=target_junction,
        leak_area=leak_orifice_area,
        discharge_coeff=0.75,
        start_time=leak_start_time,
    )
    print(f"  Injected Fault Details:")
    print(f"    Fault Type:          {leak_meta['fault_type']}")
    print(f"    Target Node:         {leak_meta['target']} (Zone: {leak_meta['zone']})")
    print(f"    Orifice Area:        {leak_meta['leak_area_m2']} m^2")
    print(f"    Equivalent Diameter: {leak_meta['equivalent_diameter_mm']} mm")
    print(f"    Discharge Coeff:     {leak_meta['discharge_coeff']}")
    print(f"    Leak Status:         {leak_meta['status']}")

    # -------------------------------------------------------------
    # Step 4: Leak Scenario Simulation
    # -------------------------------------------------------------
    print_section("4. Executing Leak Scenario Simulation")
    engine_leak = HydraulicEngine(wn_leak)
    results_leak = engine_leak.run(engine="WNTR", mode="PDD")
    print(f"  [OK] Leak Simulation Completed Successfully.")
    print(f"       Engine:              {results_leak.engine_type} (Pressure-Dependent Demand)")
    print(f"       Timesteps Computed:  {len(results_leak.pressure.index)} steps")

    # -------------------------------------------------------------
    # Step 5: Save Datasets to CSV
    # -------------------------------------------------------------
    print_section("5. Saving Simulation Datasets to CSV")
    output_dir = Path("data/simulation_results")
    saved_normal = HydraulicEngine.save_to_csv(results_normal, output_dir=str(output_dir), prefix="normal")
    saved_leak = HydraulicEngine.save_to_csv(results_leak, output_dir=str(output_dir), prefix="leak_zone_b")

    print(f"  Directory: {output_dir.resolve()}")
    print("  Baseline CSV Files:")
    for cat, filepath in saved_normal.items():
        file_size_kb = os.path.getsize(filepath) / 1024.0
        print(f"    - {Path(filepath).name:<32} ({file_size_kb:6.2f} KB)")
    print("  Leak Scenario CSV Files:")
    for cat, filepath in saved_leak.items():
        file_size_kb = os.path.getsize(filepath) / 1024.0
        print(f"    - {Path(filepath).name:<32} ({file_size_kb:6.2f} KB)")

    # -------------------------------------------------------------
    # Step 6: Detailed Comparative Analytics
    # -------------------------------------------------------------
    print_section("6. Comparative Analysis: Baseline vs. Leak")
    comparison = HydraulicEngine.compare_scenarios(
        normal=results_normal,
        leak=results_leak,
        leak_node=target_junction,
    )

    node_df = comparison["node_comparison"]
    zone_df = comparison["zone_summary"]
    sys_metrics = comparison["system_summary"]
    pipe_df = comparison["pipe_comparison"]

    print("\n[A] SYSTEM-WIDE AGGREGATED METRICS")
    print(f"  Baseline Consumer Demand:      {sys_metrics['total_normal_demand_lps']:>8.2f} L/s")
    print(f"  Leak Consumer Demand:          {sys_metrics['total_actual_consumer_demand_lps']:>8.2f} L/s")
    print(f"  Total Leakage Water Loss:      {sys_metrics['total_leak_water_loss_lps']:>8.2f} L/s")
    print(f"  Net Total Demand Increase:     {sys_metrics['system_demand_increase_pct']:>8.1f} %")
    print(f"  Maximum System Pressure Drop:  {sys_metrics['max_system_pressure_drop_m']:>8.2f} m")
    print(f"  Worst Affected Junction:       {sys_metrics['worst_affected_node']}")

    print("\n[B] ZONE-LEVEL IMPACT SUMMARY")
    print(f"{'Zone':<10} | {'Normal P (m)':<13} | {'Leak P (m)':<12} | {'Avg Drop (m)':<13} | {'Max Drop (m)':<13} | {'Leakage (L/s)':<13}")
    print("-" * 88)
    for _, row in zone_df.iterrows():
        print(
            f"{row['zone']:<10} | "
            f"{row['avg_normal_pressure_m']:>13.2f} | "
            f"{row['avg_leak_pressure_m']:>12.2f} | "
            f"{row['avg_pressure_drop_m']:>13.2f} | "
            f"{row['max_pressure_drop_m']:>13.2f} | "
            f"{row['total_leak_discharge_lps']:>13.2f}"
        )

    print("\n[C] JUNCTION-BY-JUNCTION PRESSURE & DEMAND COMPARISON")
    print(f"{'Junction':<10} | {'Zone':<8} | {'P_Norm(m)':<10} | {'P_Leak(m)':<10} | {'Drop(m)':<8} | {'Drop(%)':<8} | {'Leak(L/s)':<10} | {'Origin'}")
    print("-" * 88)
    for _, row in node_df.iterrows():
        origin_flag = "<-- LEAK SOURCE" if row["is_leak_origin"] else ""
        print(
            f"{row['junction']:<10} | "
            f"{row['zone']:<8} | "
            f"{row['normal_pressure_m']:>10.2f} | "
            f"{row['leak_pressure_m']:>10.2f} | "
            f"{row['pressure_drop_m']:>8.2f} | "
            f"{row['pressure_drop_pct']:>7.1f}% | "
            f"{row['leak_discharge_lps']:>10.2f} | "
            f"{origin_flag}"
        )

    print("\n[D] TOP PIPES BY FLOW RATE INCREASE")
    print(f"{'Pipe':<10} | {'Norm Flow(L/s)':<15} | {'Leak Flow(L/s)':<15} | {'Delta Flow(L/s)':<16} | {'Increase (%)'}")
    print("-" * 75)
    for _, row in pipe_df.head(6).iterrows():
        print(
            f"{row['pipe']:<10} | "
            f"{row['normal_flow_lps']:>15.2f} | "
            f"{row['leak_flow_lps']:>15.2f} | "
            f"{row['flow_increase_lps']:>16.2f} | "
            f"{row['flow_increase_pct']:>11.1f}%"
        )

    # -------------------------------------------------------------
    # Step 7: Automated Physical Assertions
    # -------------------------------------------------------------
    print_section("7. Physical Consistency Validations")
    leak_junction_row = node_df[node_df["junction"] == target_junction].iloc[0]
    p_norm = leak_junction_row["normal_pressure_m"]
    p_leak = leak_junction_row["leak_pressure_m"]
    q_leak = leak_junction_row["leak_discharge_lps"]

    # Assertion 1: Pressure drop at leak junction
    assert p_leak < p_norm, f"Physical Violation: Pressure at leak node did not decrease! {p_leak} >= {p_norm}"
    print(f"  [PASS] Pressure Drop Validated at {target_junction}: {p_norm:.2f} m -> {p_leak:.2f} m (-{p_norm - p_leak:.2f} m)")

    # Assertion 2: Leak discharge is strictly positive
    assert q_leak > 0.0, f"Physical Violation: Leak discharge is not positive! {q_leak}"
    print(f"  [PASS] Leak Discharge Rate Validated: {q_leak:.2f} L/s ({q_leak / 1000.0:.4f} m^3/s)")

    # Assertion 3: CSV files created and populated
    for f in list(saved_normal.values()) + list(saved_leak.values()):
        assert os.path.exists(f) and os.path.getsize(f) > 0, f"Missing or empty file: {f}"
    print(f"  [PASS] All {len(saved_normal) + len(saved_leak)} Output CSV Files Verified on Disk.")

    # Assertion 4: Zone B is the most severely affected zone
    zone_b_row = zone_df[zone_df["zone"] == "Zone_B"].iloc[0]
    assert zone_b_row["max_pressure_drop_m"] == sys_metrics["max_system_pressure_drop_m"], "Zone B should contain max pressure drop!"
    print(f"  [PASS] Localization Validated: Zone B has maximum pressure drop ({zone_b_row['max_pressure_drop_m']:.2f} m)")

    print_header("ALL SIMULATION AND FAULT INJECTION TESTS PASSED")
    return True


if __name__ == "__main__":
    success = run_test()
    if not success:
        sys.exit(1)
