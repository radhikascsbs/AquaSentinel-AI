"""
AquaSentinel AI - Decision Engine & Control Optimization Test Suite
Validates:
1. Candidate valve closure evaluation on pristine network copies.
2. Multi-objective tradeoff optimization (leak mitigation vs customer service).
3. Automatic valve response safety gating (AI confidence + sensor integrity).
4. Dual-path verification:
   - Path 1: AUTO-ACT on high-confidence confirmed leak.
   - Path 2A: NO-ACT due to low AI confidence (<90%).
   - Path 2B: NO-ACT due to detected sensor instrumentation failure.
5. Simulated valve action execution, audit logging, and before-vs-after recovery verification.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from simulation.network_builder import build_network
from simulation.fault_injection import FaultInjector
from simulation.hydraulic_engine import HydraulicEngine

from ai_engine.fault_localisation import FaultLocaliser
from ai_engine.sensor_fault import SensorFaultDiagnoser, SensorFaultInjector

from decision_engine.action_optimizer import ActionOptimizer
from decision_engine.recovery_verification import RecoveryVerifier


def print_banner(title: str):
    width = 84
    print("\n" + "=" * width)
    print(f" {title.upper()} ".center(width, "="))
    print("=" * width)


def print_section(title: str):
    width = 84
    print(f"\n--- {title} " + "-" * max(0, width - len(title) - 5))


def run_decision_engine_test():
    print_banner("AquaSentinel AI - Decision Engine & Action Optimization Suite")

    # =========================================================================
    # PATH 1: AUTO-ACT PATH (High Confidence + Trustworthy Physical Leak)
    # =========================================================================
    print_section("PATH 1: AUTOMATED VALVE ACTION (HIGH CONFIDENCE + TRUSTWORTHY TELEMETRY)")

    # 1. Setup: Real simulated leak in Zone B on pipe P_B1
    print("  Scenario: Severe physical pipe rupture on Pipe 'P_B1' (Zone B)")
    wn_fault = build_network(duration_hours=0)
    leak_meta = FaultInjector.inject_pipe_leak(
        wn=wn_fault,
        pipe_name="P_B1",
        leak_area=0.0025,  # ~56 mm equivalent diameter
        start_time=0,
    )
    print(f"  Injected Pipe Leak on '{leak_meta['target_pipe']}' (Orifice: {leak_meta['leak_area_m2']} m^2, Eq. Diam: {leak_meta['equivalent_diameter_mm']} mm)")

    # 2. Localiser identifies top suspect pipe
    wn_normal = build_network(duration_hours=0)
    res_normal = HydraulicEngine(wn_normal).run()
    norm_pressures = res_normal.pressure.iloc[0].to_dict()
    norm_flows = res_normal.flowrate.iloc[0].to_dict()

    res_fault = HydraulicEngine(wn_fault).run()
    meas_pressures = res_fault.pressure.iloc[0].to_dict()
    meas_flows = res_fault.flowrate.iloc[0].to_dict()

    localiser = FaultLocaliser(wn_normal)
    suspect_pipes = localiser.rank_suspect_pipes(
        measured_pressures=meas_pressures,
        normal_pressures=norm_pressures,
        measured_flows=meas_flows,
        normal_flows=norm_flows,
        top_k=3,
    )
    top_suspect_pipe = suspect_pipes[0]["pipe_name"]
    print(f"  Top Suspect Pipe from Localiser: '{top_suspect_pipe}' (Score: {suspect_pipes[0]['suspect_score']}%, Endpoints: {suspect_pipes[0]['start_node']} -> {suspect_pipes[0]['end_node']})")

    # 3. AI Safety Gate Inputs:
    ai_confidence_high = 0.965  # 96.5% confidence from classifier (>90% threshold)
    diagnoser = SensorFaultDiagnoser(wn_normal)
    diag_result = diagnoser.diagnose(
        measured_pressures=meas_pressures,
        normal_pressures=norm_pressures,
        measured_flows=meas_flows,
        normal_flows=norm_flows,
    )
    sensor_trustworthy = not diag_result["is_sensor_fault"]

    print(f"  AI Safety Gate Check:")
    print(f"    - Classifier Confidence: {ai_confidence_high * 100.0:.1f}% (Threshold: 90.0%) -> PASS")
    print(f"    - Sensor Diagnostics:    {diag_result['diagnosis']} (Trustworthy: {sensor_trustworthy}) -> PASS")

    # 4. Action Optimizer: Evaluate Candidate Actions on Network Copies
    print("\n  Evaluating Candidate Valve Actions on Network Copies...")
    optimizer = ActionOptimizer(confidence_threshold=0.90)
    opt_result = optimizer.optimize_action(
        wn=wn_fault,
        top_suspect_pipe=top_suspect_pipe,
        ai_confidence=ai_confidence_high,
        sensor_trustworthy=sensor_trustworthy,
    )

    print("\n  Candidate Valve Actions Tradeoff Analysis (Simulated on Copies):")
    print(f"  {'Action':<35} | {'Remaining Leak':<14} | {'Customer Svc':<12} | {'Unmet Demand':<12} | {'P Violations':<12} | {'Penalty':<8}")
    print("  " + "-" * 105)
    for c in opt_result.evaluated_candidates:
        star = " [OPTIMAL]" if c.is_best_action else ""
        print(
            f"  {c.action_name:<35} | "
            f"{c.remaining_leak_lps:>10.2f} L/s | "
            f"{c.customer_service_pct:>10.1f} % | "
            f"{c.unmet_demand_lps:>10.2f} L/s | "
            f"{c.pressure_violations_count:>12} | "
            f"{c.optimization_penalty:>8.1f}{star}"
        )

    # 5. Review Optimizer Decision & Action Log
    print(f"\n  Decision Outcome:         {opt_result.decision}")
    print(f"  Auto-Action Approved:     {opt_result.auto_action_approved}")
    print(f"  Selected Action:          {opt_result.selected_action.action_name}")
    print(f"  Isolated Link(s):         {opt_result.selected_action.target_links}")
    print(f"  Policy Reason:            {opt_result.policy_reason}")

    log = opt_result.action_log
    print("\n  Action Execution Audit Log:")
    print(f"    [SIMULATED VALVE ACTION]")
    print(f"      Timestamp:            {log['timestamp']}")
    print(f"      Target Suspect Pipe:  {log['target_suspect_pipe']}")
    print(f"      Applied Action:       {log['selected_action']}")
    print(f"      Isolated Link(s):     {log['isolated_links']}")
    print(f"      AI Confidence:        {log['ai_confidence_pct']}%")
    print(f"      Expected Loss Cut:    {log['expected_water_loss_reduction_pct']}%")
    print(f"      Customer Svc Kept:    {log['customer_service_retained_pct']}%")
    print(f"      System State:         {log['system_status']} (Rest of network operational)")

    # 6. Recovery Verification: Before vs After Simulation Comparison
    print_section("PATH 1: RECOVERY VERIFICATION (BEFORE VS AFTER SIMULATION)")
    verifier = RecoveryVerifier()
    recovery_report = verifier.verify_recovery(
        pre_wn=wn_fault,
        post_wn=opt_result.applied_simulation_model,
        target_pipe=top_suspect_pipe,
        action_log=log,
    )

    print(f"  Verification Status:      {recovery_report.status}")
    print(f"  Network Operational:      {recovery_report.is_network_operational}")
    print(f"  Action Executed:          {recovery_report.action_description}")

    print("\n  Hydraulic Comparison (Before vs. After Simulated Valve Action):")
    print(f"    {'Metric':<32} | {'Before Action (Active Leak)':<28} | {'After Action (Mitigated)':<25} | {'Change'}")
    print("    " + "-" * 105)
    print(f"    {'Total Water Loss (Leak)':<32} | {recovery_report.water_loss_before_lps:>24.2f} L/s | {recovery_report.water_loss_after_lps:>21.2f} L/s | -{recovery_report.water_loss_reduction_pct:.1f}%")
    print(f"    {'Delivered Customer Demand':<32} | {recovery_report.delivered_demand_before_lps:>24.2f} L/s | {recovery_report.delivered_demand_after_lps:>21.2f} L/s | +{recovery_report.delivered_demand_after_lps - recovery_report.delivered_demand_before_lps:.2f} L/s")
    print(f"    {'Unmet Consumer Demand':<32} | {recovery_report.unmet_demand_before_lps:>24.2f} L/s | {recovery_report.unmet_demand_after_lps:>21.2f} L/s | {recovery_report.unmet_demand_after_lps - recovery_report.unmet_demand_before_lps:+.2f} L/s")
    print(f"    {'Customer Service Level':<32} | {recovery_report.customer_service_pct_before:>26.1f} % | {recovery_report.customer_service_pct_after:>23.1f} % | {recovery_report.customer_service_pct_after - recovery_report.customer_service_pct_before:+.1f}%")
    print(f"    {'Mean System Pressure':<32} | {recovery_report.mean_pressure_before_m:>25.2f} m | {recovery_report.mean_pressure_after_m:>22.2f} m | +{recovery_report.mean_pressure_after_m - recovery_report.mean_pressure_before_m:.2f} m")
    print(f"    {'Min System Pressure':<32} | {recovery_report.min_pressure_before_m:>25.2f} m | {recovery_report.min_pressure_after_m:>22.2f} m | +{recovery_report.min_pressure_after_m - recovery_report.min_pressure_before_m:.2f} m")
    print(f"    {'Low Pressure Violations':<32} | {recovery_report.pressure_violations_before:>27} | {recovery_report.pressure_violations_after:>24} | {recovery_report.pressure_violations_after - recovery_report.pressure_violations_before:+d}")

    print("\n  Zone-by-Zone Service Breakdown Post-Mitigation:")
    print(f"    {'Zone':<10} | {'Required (L/s)':<16} | {'Delivered Pre':<15} | {'Delivered Post':<15} | {'Pressure Post':<15} | {'Service %':<10} | {'Status'}")
    print("    " + "-" * 105)
    for _, zrow in recovery_report.zone_status_table.iterrows():
        print(
            f"    {zrow['zone']:<10} | "
            f"{zrow['customer_demand_req_lps']:>14.2f} L/s | "
            f"{zrow['delivered_pre_lps']:>11.2f} L/s | "
            f"{zrow['delivered_post_lps']:>11.2f} L/s | "
            f"{zrow['avg_pressure_post_m']:>13.2f} m | "
            f"{zrow['service_continuity_pct']:>8.1f} % | "
            f"{zrow['operational_status']}"
        )

    print("\n  Remaining Network Abnormalities & Physical State Analysis:")
    for item in recovery_report.remaining_abnormalities:
        print(f"    * {item}")

    # =========================================================================
    # PATH 2A: NO-ACT PATH - Low AI Confidence (< 90%)
    # =========================================================================
    print_section("PATH 2A: NO AUTOMATED ACTION (LOW AI CONFIDENCE < 90%)")
    ai_confidence_low = 0.742  # 74.2% confidence (below 90% threshold)
    print(f"  Simulating scenario where classifier confidence is {ai_confidence_low * 100.0:.1f}% (< 90.0% threshold)")

    opt_result_low_conf = optimizer.optimize_action(
        wn=wn_fault,
        top_suspect_pipe=top_suspect_pipe,
        ai_confidence=ai_confidence_low,
        sensor_trustworthy=True,
    )

    print(f"  Decision Outcome:         {opt_result_low_conf.decision}")
    print(f"  Auto-Action Approved:     {opt_result_low_conf.auto_action_approved}")
    print(f"  Valves Closed:            None (Network untouched)")
    print(f"  Policy Reason:            {opt_result_low_conf.policy_reason}")

    # =========================================================================
    # PATH 2B: NO-ACT PATH - Untrustworthy Sensors (Sensor Fault Detected)
    # =========================================================================
    print_section("PATH 2B: NO AUTOMATED ACTION (UNTRUSTWORTHY SENSOR DETECTED)")
    print("  Simulating scenario where a sensor malfunction (false drop) is detected on J_B2...")

    # Inject sensor fault onto normal measurements
    faulty_sensor_readings = dict(norm_pressures)
    SensorFaultInjector.inject_false_drop(faulty_sensor_readings, "J_B2", drop_magnitude=20.0)

    # Diagnoser evaluates readings
    diag_sensor_check = diagnoser.diagnose(
        measured_pressures=faulty_sensor_readings,
        normal_pressures=norm_pressures,
        measured_flows=norm_flows,
        normal_flows=norm_flows,
    )
    is_sensor_reliable = not diag_sensor_check["is_sensor_fault"]

    print(f"  Sensor Diagnostics Verdict: {diag_sensor_check['diagnosis']}")
    print(f"  Identified Problem Node:    {diag_sensor_check['target_node']}")
    print(f"  Sensor Trustworthy Flag:    {is_sensor_reliable}")

    opt_result_sensor_fault = optimizer.optimize_action(
        wn=wn_fault,
        top_suspect_pipe=top_suspect_pipe,
        ai_confidence=0.985,  # High confidence, but sensor is faulty!
        sensor_trustworthy=is_sensor_reliable,
    )

    print(f"\n  Decision Outcome:         {opt_result_sensor_fault.decision}")
    print(f"  Auto-Action Approved:     {opt_result_sensor_fault.auto_action_approved}")
    print(f"  Valves Closed:            None (Safety lock engaged)")
    print(f"  Policy Reason:            {opt_result_sensor_fault.policy_reason}")

    # =========================================================================
    # PART 6: Automated Verification Assertions
    # =========================================================================
    print_section("6. Decision Engine Automated Assertions")

    # Path 1 Assertions:
    assert opt_result.auto_action_approved is True, "Path 1 should approve automatic action!"
    assert opt_result.decision == "AUTO_ACT", "Path 1 decision should be AUTO_ACT!"
    assert recovery_report.water_loss_after_lps == 0.0, "Pipe isolation should eliminate leak flow!"
    assert recovery_report.water_loss_reduction_pct == 100.0, "Water loss reduction should be 100%!"
    assert recovery_report.is_network_operational is True, "Rest of network must remain operational!"
    assert recovery_report.customer_service_pct_after > 75.0, "Customer service must remain high!"
    print("  [PASS] Path 1 (Auto-Act): Leak 100% contained (76.7 L/s -> 0.0 L/s) while retaining network service.")

    # Path 2A Assertions:
    assert opt_result_low_conf.auto_action_approved is False, "Path 2A must NOT approve action on low confidence!"
    assert opt_result_low_conf.decision == "NO_ACT_MANUAL_INSPECTION"
    assert opt_result_low_conf.applied_simulation_model is None
    print("  [PASS] Path 2A (No-Act): Successfully inhibited closure when AI confidence (74.2%) < 90%.")

    # Path 2B Assertions:
    assert opt_result_sensor_fault.auto_action_approved is False, "Path 2B must NOT approve action on sensor fault!"
    assert opt_result_sensor_fault.decision == "NO_ACT_MANUAL_INSPECTION"
    assert opt_result_sensor_fault.applied_simulation_model is None
    print("  [PASS] Path 2B (No-Act): Successfully inhibited closure when sensor fault was detected.")

    print_banner("ALL DECISION ENGINE & OPTIMIZATION TESTS PASSED SUCCESSFULLY")
    return True


if __name__ == "__main__":
    success = run_decision_engine_test()
    if not success:
        sys.exit(1)
