# AquaSentinel AI

AquaSentinel AI is an autonomous cyber-physical digital twin and decision intelligence platform for municipal water distribution networks (WDNs). It couples high-fidelity hydraulic simulation (using **WNTR** / **EPANET**) with machine learning anomaly detection, physics-informed topological fault localisation, sensor integrity validation, and automated valve response optimization.

---

## Key System Capabilities

- **Physics-Based Hydraulic Digital Twin**: Built on EPA-backed hydraulic solvers (**WNTR 1.5.0** Pressure-Dependent Demand / PDD and Demand-Driven simulation).
- **Multi-Zone Municipal Network**: 1 Reservoir, 15 Junctions across 3 District Metered Areas (DMA Zones A, B, C), 17 Pipes, 2 Multi-Stage Pumps, and 3 Motorized Isolation Valves.
- **Machine Learning Anomaly Detection**: Isolation Forest trained exclusively on pristine baseline operating telemetry (ROC AUC = 1.000).
- **Supervised Fault Classification**: Random Forest classifier discriminating normal vs burst leak regimes (Accuracy = 100%, F1 = 100%).
- **Topological Fault Localisation**: NetworkX graph theory and pressure-gradient field analysis ranking top-3 suspect pipes with likelihood scores.
- **Physics-Informed Sensor Fault Discrimination**: Divergence analysis distinguishing physical pipe bursts from instrumentation failures (stuck, drifting, or false-drop sensors).
- **Autonomous Decision Engine**: Evaluates candidate valve closures on in-memory network copies, optimizes leak mitigation vs. customer service continuity, and enforces strict safety gating (AI confidence $\ge 90\%$ AND verified sensor integrity).
- **Before-vs-After Recovery Verification**: Re-simulates post-mitigation networks to verify 100% loss containment, customer service retention, and pressure stabilization.
- **Interactive Streamlit Dashboard**: Dark navy industrial UI featuring real-time telemetry KPI cards, interactive Plotly network twin, simulation lab, and AI decision console.

---

## Architecture Overview

```
AquaSentinel-AI/
├── simulation/
│   ├── network_builder.py          # 15-node, 17-pipe, 3-zone municipal WDN model
│   ├── hydraulic_engine.py         # WNTR PDD simulation engine and metrics exporter
│   └── fault_injection.py          # Physical orifice leak and pipe-splitting injectors
├── ai_engine/
│   ├── data_generator.py           # Multi-scenario generator with stochastic demand [80-125%]
│   ├── anomaly_detection.py        # Unsupervised Isolation Forest (trained on normal only)
│   ├── fault_classifier.py         # Supervised Random Forest fault classifier
│   ├── fault_localisation.py       # NetworkX topological pressure-gradient ranker
│   └── sensor_fault.py             # Sensor integrity vs physical leak diagnoser
├── decision_engine/
│   ├── action_optimizer.py         # Valve closure evaluator and safety policy governor
│   └── recovery_verification.py    # Pre/post hydraulic recovery and service verifier
├── results/                        # QA experiment CSV datasets and JSON execution logs
│   ├── normal_baseline.csv
│   ├── zone_b_leak_experiment.csv
│   ├── valve_action_recovery.csv
│   ├── sensor_fault_experiment.csv
│   └── qa_test_report.json
├── test_simulation.py              # WNTR hydraulic engine test
├── test_ai_pipeline.py             # AI & ML pipeline test
├── test_decision_pipeline.py       # Decision optimizer & recovery test
├── test_app_qa.py                  # End-to-end 5-point QA test suite
├── app.py                          # 4-page Streamlit digital twin dashboard
└── requirements.txt                # Pinned production dependencies
```

---

## Installation & Setup

### Prerequisites
- **Python 3.10+** (tested on Python 3.11.9 64-bit on Windows and Linux)
- Git

### 1. Clone or Open the Workspace
```powershell
cd AquaSentinel-AI
```

### 2. Create and Activate Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Linux / macOS (bash/zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## Running the Application

### Single Launch Command
Run the Streamlit digital twin dashboard directly from your terminal:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
*(Or simply `streamlit run app.py` if your virtual environment is already activated).*

Open your browser to:
```
http://localhost:8501
```

---

## Dashboard Pages & Features

### 1. Overview
- **Real-Time KPIs**: Total Supply Flow (L/s), Mean System Pressure (m), Active Incident Status, AI Health Index.
- **Hydraulic Telemetry Table**: Real-time junction pressures, head elevations, consumer demands, and leak rates.

### 2. Digital Twin
- **Interactive Plotly Graph**: Topological layout of the municipal network across Zones A, B, and C.
- **Dynamic Color Coding**:
  - Green nodes: Junctions operating at normal pressure ($\ge 15.0$ m).
  - Amber/Red nodes: Junctions experiencing pressure depression.
  - Cyan lines: Normal operating transmission pipes.
  - Red pulsing lines: Suspected leak rupture pipe.
  - Yellow dashed lines: Closed/isolated valve segments.
- **Hover Telemetry**: Inspect exact nodal pressure, elevation, and link flow rates.

### 3. Simulation Lab
- **Zone B Leak Injection**: Select target pipe segment (`P_B1`, `P_B2`, `P_B3`, `P_B4`, `P_B5`, `P_B6`) and leak severity (Minor, Moderate, Severe Rupture).
- **False Alarm Testing**: Check **"Inject Faulty Sensor Reading"** to simulate an isolated sensor failure (sudden false drop, sensor drift, or stuck reading) and verify that the safety governor inhibits accidental valve closure.
- **Interactive Execution**: Click **"RUN LEAK SIMULATION"** to run the live WNTR hydraulic solver.

### 4. AI & Response
- **AI Diagnostics Panel**: Continuous anomaly likelihood score [0.0 - 1.0], Random Forest classification verdict, and confidence percentage.
- **Top-3 Suspect Pipe Ranking**: NetworkX proximity and hydraulic gradient breakdown.
- **Sensor Health Telemetry**: Live status for every junction sensor (`OK` vs `FAULTY`).
- **Autonomous Response Engine**:
  - Automatically isolates the rupture segment when AI confidence $\ge 90\%$ and sensors are trustworthy.
  - Inhibits automated actions and logs safety warnings if confidence is low or a sensor is faulty.
- **Manual Override Execution**: Click **"EXECUTE ACTION"** to manually force valve actuation.
- **Before vs. After Recovery Visuals**: Side-by-side charts illustrating leak containment, customer service retention, and pressure stabilization.

---

## Running the Test Suites

To verify any component independently, run the targeted test suites:

### 1. Hydraulic Engine Verification
```powershell
.\.venv\Scripts\python.exe test_simulation.py
```
*Validates normal 24h baseline simulation, burst leak injection, and CSV export.*

### 2. Machine Learning & Sensor Fault Pipeline
```powershell
.\.venv\Scripts\python.exe test_ai_pipeline.py
```
*Validates Isolation Forest ROC AUC, Random Forest classifier on held-out test data, NetworkX fault localisation, and physics-informed sensor fault discrimination.*

### 3. Decision Engine & Action Optimizer
```powershell
.\.venv\Scripts\python.exe test_decision_pipeline.py
```
*Validates automated valve closure tradeoffs, recovery verification metrics, and low-confidence/sensor-fault safety gating.*

### 4. Final Comprehensive End-to-End QA
```powershell
.\.venv\Scripts\python.exe test_app_qa.py
```
*Executes the complete 5-point QA test suite and exports verified datasets to `results/`.*

---

## Experiment Results Summary

The automated QA suite produces verified experiment outputs stored in `results/`:

| Experiment File | Core Verification | Outcome |
| :--- | :--- | :--- |
| `results/normal_baseline.csv` | Baseline network without anomalies | Anomaly score: `0.0843` (Normal), Leak: `0.00 L/s`, Pressure: `98.09 m` |
| `results/zone_b_leak_experiment.csv` | Physical burst leak on pipe `P_B1` | Rupture: `76.73 L/s`, Conf: `100.0%`, Rank #1: `P_B1` (Score: `100.0%`) |
| `results/valve_action_recovery.csv` | Mitigation of burst leak via valve closure | Leak reduced from `77.58 L/s` to `0.00 L/s` (**-100.0%**), Service: **109.9%** |
| `results/sensor_fault_experiment.csv` | False drop (-18m) injected on `J_B2` sensor | Verdict: `SENSOR_FAULT`, Decision: `NO_ACT_MANUAL_INSPECTION`, Zero valves closed |
| `results/qa_test_report.json` | Complete programmatic validation JSON | `ALL_TESTS_PASSED` (5/5 tests passed) |
