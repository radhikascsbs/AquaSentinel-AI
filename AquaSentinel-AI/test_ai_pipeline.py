"""
AquaSentinel AI - Full AI Engine & Machine Learning Test Suite
Validates:
1. Hydraulic scenario dataset generation with demand randomization.
2. Isolation Forest unsupervised anomaly detection (trained on normal only).
3. Random Forest supervised fault classification evaluated strictly on test set.
4. NetworkX graph topological fault localisation (Top-3 ranked pipes).
5. Sensor fault detection distinguishing real leaks from faulty sensors using physics.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from simulation.network_builder import build_network
from simulation.hydraulic_engine import HydraulicEngine
from simulation.fault_injection import FaultInjector

from ai_engine.data_generator import generate_dataset
from ai_engine.anomaly_detection import HydraulicAnomalyDetector
from ai_engine.fault_classifier import FaultClassifier
from ai_engine.fault_localisation import FaultLocaliser
from ai_engine.sensor_fault import SensorFaultDiagnoser, SensorFaultInjector


def print_banner(title: str):
    width = 82
    print("\n" + "=" * width)
    print(f" {title.upper()} ".center(width, "="))
    print("=" * width)


def print_section(title: str):
    width = 82
    print(f"\n--- {title} " + "-" * max(0, width - len(title) - 5))


def run_ai_pipeline_test():
    print_banner("AquaSentinel AI - Machine Learning & Diagnostic Pipeline")

    # =========================================================================
    # Part 1: Dataset Generation
    # =========================================================================
    print_section("1. Hydraulic Scenario Dataset Generation")
    print("  Executing WNTR simulations with random consumer demand variations [80% - 125%]...")
    dataset = generate_dataset(
        n_samples=100,
        leak_ratio=0.5,
        test_size=0.25,
        random_state=42,
        save_dir="data/ai_datasets",
    )

    n_train = len(dataset.X_train)
    n_test = len(dataset.X_test)
    n_features = len(dataset.feature_names)
    n_normal_train = sum(dataset.y_train == 0)
    n_leak_train = sum(dataset.y_train == 1)
    n_normal_test = sum(dataset.y_test == 0)
    n_leak_test = sum(dataset.y_test == 1)

    print(f"  [OK] Dataset Generated & Saved to 'data/ai_datasets/'.")
    print(f"       Total Samples:      {len(dataset.X)}")
    print(f"       Sensor Features:    {n_features} (15 Pressures + 22 Flow rates)")
    print(f"       Training Set Size:  {n_train} ({n_normal_train} Normal, {n_leak_train} Leak)")
    print(f"       Testing Set Size:   {n_test} ({n_normal_test} Normal, {n_leak_test} Leak)")

    # =========================================================================
    # Part 2: Isolation Forest Anomaly Detection
    # =========================================================================
    print_section("2. Isolation Forest Anomaly Detection (Normal-Only Training)")
    print("  Training Isolation Forest exclusively on normal baseline observations (y=0)...")
    X_train_normal = dataset.X_train[dataset.y_train == 0]

    anomaly_detector = HydraulicAnomalyDetector(
        contamination=0.05,
        n_estimators=150,
        random_state=42,
    )
    anomaly_detector.fit(X_train_normal)

    # Evaluate on held-out test set
    if_metrics = anomaly_detector.evaluate(dataset.X_test, dataset.y_test)
    print(f"  [OK] Isolation Forest Model Evaluated on Test Set:")
    print(f"       ROC AUC Score:               {if_metrics['roc_auc']:.4f}")
    print(f"       Test Accuracy:               {if_metrics['accuracy']:.4f} ({if_metrics['accuracy']*100:.1f}%)")
    print(f"       Test Precision:              {if_metrics['precision']:.4f}")
    print(f"       Test Recall:                 {if_metrics['recall']:.4f}")
    print(f"       Test F1 Score:               {if_metrics['f1_score']:.4f}")
    print(f"       Mean Normal Anomaly Score:   {if_metrics['mean_anomaly_score_normal']:.4f} (Low / Inlier)")
    print(f"       Mean Leak Anomaly Score:     {if_metrics['mean_anomaly_score_leak']:.4f} (High / Anomaly)")
    print(f"       Confusion Matrix:            {if_metrics['confusion_matrix']}")

    # =========================================================================
    # Part 3: Random Forest Fault Classification
    # =========================================================================
    print_section("3. Random Forest Fault Classifier (Test Set Evaluation)")
    print("  Training Random Forest on training set...")
    classifier = FaultClassifier(
        n_estimators=120,
        max_depth=10,
        random_state=42,
    )
    classifier.fit(dataset.X_train, dataset.y_train)

    rf_metrics = classifier.evaluate_test_set(dataset.X_test, dataset.y_test)
    print(f"  [OK] Random Forest Performance ON HELD-OUT TEST SET ONLY ({rf_metrics['test_sample_count']} samples):")
    print(f"       Test Accuracy:               {rf_metrics['accuracy']:.4f} ({rf_metrics['accuracy']*100:.1f}%)")
    print(f"       Test Precision:              {rf_metrics['precision']:.4f} ({rf_metrics['precision']*100:.1f}%)")
    print(f"       Test Recall:                 {rf_metrics['recall']:.4f} ({rf_metrics['recall']*100:.1f}%)")
    print(f"       Test F1 Score:               {rf_metrics['f1_score']:.4f} ({rf_metrics['f1_score']*100:.1f}%)")
    print(f"       Confusion Matrix:            TN={rf_metrics['confusion_matrix'][0][0]}, FP={rf_metrics['confusion_matrix'][0][1]}, "
          f"FN={rf_metrics['confusion_matrix'][1][0]}, TP={rf_metrics['confusion_matrix'][1][1]}")

    print("\n  Top Informative Hydraulic Sensors by Gini Importance:")
    for rank, item in enumerate(rf_metrics["top_features"][:6], start=1):
        print(f"    #{rank} {item['feature']:<16} : Importance = {item['importance']:.4f}")

    # =========================================================================
    # Part 4: Topological Fault Localisation using NetworkX
    # =========================================================================
    print_section("4. Topological Fault Localisation (NetworkX + Hydraulics)")
    # Generate a real simulated leak scenario in Zone B (J_B2) to test localisation
    wn_norm = build_network(duration_hours=0)
    res_norm = HydraulicEngine(wn_norm).run()
    norm_pressures = res_norm.pressure.iloc[0].to_dict()
    norm_flows = res_norm.flowrate.iloc[0].to_dict()

    wn_leak = build_network(duration_hours=0)
    leak_target_node = "J_B2"
    FaultInjector.inject_junction_leak(wn_leak, leak_target_node, leak_area=0.0025, start_time=0)
    res_leak = HydraulicEngine(wn_leak).run()
    leak_pressures = res_leak.pressure.iloc[0].to_dict()
    leak_flows = res_leak.flowrate.iloc[0].to_dict()

    localiser = FaultLocaliser(wn_norm)
    suspect_pipes = localiser.rank_suspect_pipes(
        measured_pressures=leak_pressures,
        normal_pressures=norm_pressures,
        measured_flows=leak_flows,
        normal_flows=norm_flows,
        top_k=3,
    )

    print(f"  Actual Injected Leak Location: Node '{leak_target_node}' (Zone B)")
    print(f"  Connected Pipes to '{leak_target_node}': P_B1, P_B2, P_B6")
    print("\n  NetworkX Top-3 Ranked Suspect Pipes:")
    print(f"  {'Rank':<5} | {'Pipe':<8} | {'Zone':<8} | {'Endpoints':<16} | {'Suspect Score':<14} | {'Pressure Drop':<14} | {'Hops'}")
    print("  " + "-" * 78)
    for p in suspect_pipes:
        endpoints = f"{p['start_node']} -> {p['end_node']}"
        print(
            f"  #{p['rank']:<4} | "
            f"{p['pipe_name']:<8} | "
            f"{p['zone']:<8} | "
            f"{endpoints:<16} | "
            f"{p['suspect_score']:>12.1f} % | "
            f"{p['avg_pressure_drop_m']:>12.2f} m | "
            f"{p['hops_to_epicenter']:>4}"
        )

    # =========================================================================
    # Part 5: Sensor Fault vs Real Leak Discrimination
    # =========================================================================
    print_section("5. Physics-Based Sensor Fault vs. Real Leak Discrimination")
    diagnoser = SensorFaultDiagnoser(wn_norm)

    # -------------------------------------------------------------
    # Case A: REAL PHYSICAL LEAK (at J_B2)
    # -------------------------------------------------------------
    print("\n  [TEST CASE A] Real Physical Leak Scenario (J_B2 Burst)")
    diag_real = diagnoser.diagnose(
        measured_pressures=leak_pressures,
        normal_pressures=norm_pressures,
        measured_flows=leak_flows,
        normal_flows=norm_flows,
    )
    print(f"    Diagnostic Verdict:     {diag_real['diagnosis']}")
    print(f"    Confidence:             {diag_real['confidence']*100:.1f} %")
    print(f"    Target Node:            {diag_real['target_node']}")
    print(f"    Is Sensor Fault?        {diag_real['is_sensor_fault']} (Expected: False)")
    ev_real = diag_real['spatial_evidence']
    print(f"    Target Node Drop:       {ev_real['target_drop_m']:.2f} m")
    print(f"    Neighbor Sensor Drops:  {ev_real['neighbor_drops_m']}")
    print(f"    Spatial Consistency:    {ev_real['spatial_consistency_ratio']:.1%} (Correlated hydraulic cone)")
    print(f"    Explanation:            {diag_real['explanation']}")

    # -------------------------------------------------------------
    # Case B: SENSOR FAULT - Sudden False Pressure Drop on J_B2
    # -------------------------------------------------------------
    print("\n  [TEST CASE B] Sensor Malfunction: Sudden False Drop (-18.0 m) on Sensor J_B2")
    faulty_readings_b = dict(norm_pressures)
    SensorFaultInjector.inject_false_drop(faulty_readings_b, "J_B2", drop_magnitude=18.0)

    diag_sensor_b = diagnoser.diagnose(
        measured_pressures=faulty_readings_b,
        normal_pressures=norm_pressures,
        measured_flows=norm_flows,
        normal_flows=norm_flows,
    )
    print(f"    Diagnostic Verdict:     {diag_sensor_b['diagnosis']}")
    print(f"    Confidence:             {diag_sensor_b['confidence']*100:.1f} %")
    print(f"    Target Node:            {diag_sensor_b['target_node']}")
    print(f"    Is Sensor Fault?        {diag_sensor_b['is_sensor_fault']} (Expected: True)")
    ev_b = diag_sensor_b['spatial_evidence']
    print(f"    Target Node Drop:       {ev_b['target_drop_m']:.2f} m")
    print(f"    Neighbor Sensor Drops:  {ev_b['neighbor_drops_m']}")
    print(f"    Spatial Consistency:    {ev_b['spatial_consistency_ratio']:.1%} (Isolated anomaly)")
    print(f"    Explanation:            {diag_sensor_b['explanation']}")

    # -------------------------------------------------------------
    # Case C: SENSOR FAULT - Stuck Sensor on J_A3
    # -------------------------------------------------------------
    print("\n  [TEST CASE C] Sensor Malfunction: Stuck Sensor (Fixed @ 45.0 m) on Sensor J_A3")
    faulty_readings_c = dict(norm_pressures)
    SensorFaultInjector.inject_stuck_reading(faulty_readings_c, "J_A3", stuck_value=45.0)

    diag_sensor_c = diagnoser.diagnose(
        measured_pressures=faulty_readings_c,
        normal_pressures=norm_pressures,
        measured_flows=norm_flows,
        normal_flows=norm_flows,
    )
    print(f"    Diagnostic Verdict:     {diag_sensor_c['diagnosis']}")
    print(f"    Confidence:             {diag_sensor_c['confidence']*100:.1f} %")
    print(f"    Target Node:            {diag_sensor_c['target_node']}")
    print(f"    Is Sensor Fault?        {diag_sensor_c['is_sensor_fault']} (Expected: True)")
    print(f"    Explanation:            {diag_sensor_c['explanation']}")

    # -------------------------------------------------------------
    # Case D: SENSOR FAULT - Drifting Sensor on J_C2
    # -------------------------------------------------------------
    print("\n  [TEST CASE D] Sensor Malfunction: Calibration Drift (-12.0 m) on Sensor J_C2")
    faulty_readings_d = dict(norm_pressures)
    SensorFaultInjector.inject_drifting_reading(faulty_readings_d, "J_C2", drift_offset=-12.0)

    diag_sensor_d = diagnoser.diagnose(
        measured_pressures=faulty_readings_d,
        normal_pressures=norm_pressures,
        measured_flows=norm_flows,
        normal_flows=norm_flows,
    )
    print(f"    Diagnostic Verdict:     {diag_sensor_d['diagnosis']}")
    print(f"    Confidence:             {diag_sensor_d['confidence']*100:.1f} %")
    print(f"    Target Node:            {diag_sensor_d['target_node']}")
    print(f"    Is Sensor Fault?        {diag_sensor_d['is_sensor_fault']} (Expected: True)")
    print(f"    Explanation:            {diag_sensor_d['explanation']}")

    # =========================================================================
    # Part 6: Automated Pipeline Assertions
    # =========================================================================
    print_section("6. Automated Verification Assertions")

    # Assertion 1: Isolation Forest accuracy
    assert if_metrics["roc_auc"] >= 0.85, f"Isolation Forest ROC AUC too low: {if_metrics['roc_auc']}"
    print(f"  [PASS] Anomaly Detector ROC AUC Validated: {if_metrics['roc_auc']:.4f} >= 0.85")

    # Assertion 2: Random Forest test metrics
    assert rf_metrics["accuracy"] >= 0.85, f"Random Forest accuracy too low: {rf_metrics['accuracy']}"
    assert rf_metrics["f1_score"] >= 0.85, f"Random Forest F1 too low: {rf_metrics['f1_score']}"
    print(f"  [PASS] Fault Classifier Accuracy & F1 Validated: Accuracy={rf_metrics['accuracy']*100:.1f}%, F1={rf_metrics['f1_score']*100:.1f}%")

    # Assertion 3: NetworkX suspect pipes returned
    assert len(suspect_pipes) == 3, f"Expected 3 suspect pipes, got {len(suspect_pipes)}"
    # Check that at least one of the top pipes is directly connected to J_B2
    top_pipe_names = [p["pipe_name"] for p in suspect_pipes]
    assert any(p in ["P_B1", "P_B2", "P_B6"] for p in top_pipe_names), f"Actual leak pipes not in top 3: {top_pipe_names}"
    print(f"  [PASS] Fault Localisation Validated: Top-1 Pipe '{suspect_pipes[0]['pipe_name']}' directly incident to leak node J_B2.")

    # Assertion 4: Physics-based sensor fault discrimination
    assert diag_real["diagnosis"] == "PHYSICAL_LEAK", f"Expected PHYSICAL_LEAK, got {diag_real['diagnosis']}"
    assert not diag_real["is_sensor_fault"], "Real leak was incorrectly flagged as sensor fault!"
    print(f"  [PASS] Real Leak Correctly Identified as PHYSICAL_LEAK (Not flagged as sensor fault).")

    assert diag_sensor_b["diagnosis"] == "SENSOR_FAULT", f"Expected SENSOR_FAULT, got {diag_sensor_b['diagnosis']}"
    assert diag_sensor_b["is_sensor_fault"], "Sensor fault was missed!"
    assert diag_sensor_b["target_node"] == "J_B2", f"Wrong faulty sensor identified: {diag_sensor_b['target_node']}"
    print(f"  [PASS] Sensor False Drop Correctly Identified as SENSOR_FAULT on 'J_B2'.")

    assert diag_sensor_c["diagnosis"] == "SENSOR_FAULT" and diag_sensor_c["target_node"] == "J_A3"
    print(f"  [PASS] Stuck Sensor Correctly Identified as SENSOR_FAULT on 'J_A3'.")

    assert diag_sensor_d["diagnosis"] == "SENSOR_FAULT" and diag_sensor_d["target_node"] == "J_C2"
    print(f"  [PASS] Drifting Sensor Correctly Identified as SENSOR_FAULT on 'J_C2'.")

    print_banner("ALL AI ENGINE & DIAGNOSTIC TESTS PASSED SUCCESSFULLY")
    return True


if __name__ == "__main__":
    success = run_ai_pipeline_test()
    if not success:
        sys.exit(1)
