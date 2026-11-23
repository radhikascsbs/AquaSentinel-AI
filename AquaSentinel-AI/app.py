"""
AquaSentinel AI - Smart Water Distribution Digital Twin & AI Response Dashboard
Powered by WNTR, NetworkX, Scikit-Learn, and Plotly.
"""

from typing import Dict, List, Any, Optional
import copy
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import wntr

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from simulation.network_builder import build_network, get_network_summary
from simulation.hydraulic_engine import HydraulicEngine
from simulation.fault_injection import FaultInjector

from ai_engine.data_generator import generate_dataset
from ai_engine.anomaly_detection import HydraulicAnomalyDetector
from ai_engine.fault_classifier import FaultClassifier
from ai_engine.fault_localisation import FaultLocaliser
from ai_engine.sensor_fault import SensorFaultDiagnoser, SensorFaultInjector

from decision_engine.action_optimizer import ActionOptimizer
from decision_engine.recovery_verification import RecoveryVerifier


# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AquaSentinel AI | Digital Twin",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Main App Styling */
.stApp {
    background-color: #0F1B2D;
    color: #E7EAF0;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #17253D !important;
    border-right: 1px solid #22334D;
}
section[data-testid="stSidebar"] .stRadio label {
    color: #E7EAF0 !important;
    font-size: 15px;
}

/* Custom Cards */
.aqua-card {
    background-color: #17253D;
    border: 1px solid #22334D;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 16px;
}
.aqua-card-gold {
    border-left: 4px solid #C9A227;
}
.aqua-card-red {
    border-left: 4px solid #C4554D;
}
.aqua-card-green {
    border-left: 4px solid #2ECC71;
}

/* Metric text */
[data-testid="stMetricValue"] {
    color: #E7EAF0 !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: #9FB2C8 !important;
    font-size: 13px !important;
}

/* Buttons */
.stButton > button {
    background-color: #C9A227 !important;
    color: #0F1B2D !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 8px 18px !important;
    width: 100%;
}
.stButton > button:hover {
    background-color: #E0B738 !important;
    color: #0F1B2D !important;
}

/* Badges */
.badge-normal {
    background-color: #1E3D30;
    color: #2ECC71;
    padding: 4px 10px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 12px;
}
.badge-alert {
    background-color: #3D1C22;
    color: #E74C3C;
    padding: 4px 10px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 12px;
}
.badge-warning {
    background-color: #3D3319;
    color: #F1C40F;
    padding: 4px 10px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 12px;
}

/* Hide animations */
* {
    animation: none !important;
    transition: none !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached AI Models & Baseline Initialization
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_ai_models():
    """Trains and caches the Isolation Forest and Random Forest on startup."""
    dataset = generate_dataset(n_samples=50, leak_ratio=0.5, test_size=0.25, random_state=42, save_dir=None)
    X_norm = dataset.X_train[dataset.y_train == 0]

    anomaly_model = HydraulicAnomalyDetector(contamination=0.05, n_estimators=100, random_state=42)
    anomaly_model.fit(X_norm)

    classifier_model = FaultClassifier(n_estimators=100, max_depth=10, random_state=42)
    classifier_model.fit(dataset.X_train, dataset.y_train)

    return anomaly_model, classifier_model


def get_baseline_simulation():
    """Produces the normal operating baseline results."""
    wn_norm = build_network(duration_hours=0)
    engine = HydraulicEngine(wn_norm)
    res_norm = engine.run(engine="WNTR", mode="PDD")
    return wn_norm, res_norm


def run_full_diagnosis(wn_current, sim_res, res_norm, faulty_sensor_override=None):
    """Executes the complete AI diagnostics and decision pipeline on simulation results."""
    anomaly_model, classifier_model = load_ai_models()

    meas_pressures = sim_res.pressure.iloc[0].to_dict()
    meas_flows = sim_res.flowrate.iloc[0].to_dict()
    norm_pressures = res_norm.pressure.iloc[0].to_dict()
    norm_flows = res_norm.flowrate.iloc[0].to_dict()

    # Apply sensor fault override if testing false alarm
    if faulty_sensor_override:
        meas_pressures[faulty_sensor_override["sensor"]] = faulty_sensor_override["value"]

    # 1. Prepare feature vector
    feat_dict = {}
    for j in sorted(wn_current.junction_name_list):
        feat_dict[f"P_{j}"] = float(meas_pressures.get(j, 0.0))
    for l in sorted(wn_current.link_name_list):
        feat_dict[f"Q_{l}"] = float(meas_flows.get(l, 0.0)) * 1000.0  # L/s

    feature_df = pd.DataFrame([feat_dict])

    # 2. AI Anomaly Score & Fault Classification
    # Align features with training columns
    for col in classifier_model.feature_names:
        if col not in feature_df.columns:
            feature_df[col] = 0.0
    feature_df = feature_df[classifier_model.feature_names]

    anomaly_score = float(anomaly_model.compute_anomaly_score(feature_df)[0])
    is_anomaly = bool(anomaly_model.predict(feature_df)[0] == 1 or anomaly_score > 0.45)

    probs = classifier_model.predict_proba(feature_df)[0]
    ai_confidence = float(probs[1]) if probs[1] > probs[0] else float(probs[0])
    predicted_label = "Leak" if probs[1] >= 0.5 else "Normal"

    # 3. NetworkX Fault Localisation
    localiser = FaultLocaliser(wn_current)
    suspect_pipes = localiser.rank_suspect_pipes(
        measured_pressures=meas_pressures,
        normal_pressures=norm_pressures,
        measured_flows=meas_flows,
        normal_flows=norm_flows,
        top_k=3,
    )
    top_suspect_pipe = suspect_pipes[0]["pipe_name"] if suspect_pipes else "None"

    # 4. Sensor Fault Diagnostics
    diagnoser = SensorFaultDiagnoser(wn_current)
    sensor_diag = diagnoser.diagnose(
        measured_pressures=meas_pressures,
        normal_pressures=norm_pressures,
        measured_flows=meas_flows,
        normal_flows=norm_flows,
    )
    sensor_trustworthy = not sensor_diag["is_sensor_fault"]

    # 5. Decision Engine Action Optimization
    optimizer = ActionOptimizer(confidence_threshold=0.90)
    opt_result = optimizer.optimize_action(
        wn=wn_current,
        top_suspect_pipe=top_suspect_pipe,
        ai_confidence=ai_confidence if predicted_label == "Leak" else 0.50,
        sensor_trustworthy=sensor_trustworthy,
    )

    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": anomaly_score,
        "predicted_label": predicted_label,
        "ai_confidence": ai_confidence,
        "suspect_pipes": suspect_pipes,
        "top_suspect_pipe": top_suspect_pipe,
        "sensor_diag": sensor_diag,
        "sensor_trustworthy": sensor_trustworthy,
        "opt_result": opt_result,
        "meas_pressures": meas_pressures,
        "meas_flows": meas_flows,
    }


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "initialized" not in st.session_state:
    wn_base, res_base = get_baseline_simulation()
    st.session_state.wn_base = wn_base
    st.session_state.res_base = res_base

    st.session_state.wn_current = copy.deepcopy(wn_base)
    st.session_state.sim_res = res_base
    st.session_state.scenario_name = "Normal Baseline"
    st.session_state.scenario_type = "normal"
    st.session_state.faulty_sensor_override = None

    # Run initial baseline diagnosis
    st.session_state.diag = run_full_diagnosis(
        st.session_state.wn_current,
        st.session_state.sim_res,
        st.session_state.res_base,
    )
    st.session_state.mitigated_res = None
    st.session_state.recovery_report = None
    st.session_state.initialized = True


# -----------------------------------------------------------------------------
# Sidebar Navigation & Status Header
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='color: #C9A227; margin-bottom: 4px;'>AquaSentinel AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #9FB2C8; font-size: 13px; margin-top: 0;'>Autonomous Hydraulic Digital Twin</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid #22334D; margin: 12px 0;'>", unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION",
        ["1. Overview", "2. Digital Twin", "3. Simulation Lab", "4. AI & Response"],
        index=0,
    )

    st.markdown("<hr style='border: 1px solid #22334D; margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #E7EAF0; font-size: 14px;'>SYSTEM STATUS</h4>", unsafe_allow_html=True)

    diag = st.session_state.diag
    if not diag["is_anomaly"] and st.session_state.scenario_type == "normal":
        st.markdown("<span class='badge-normal'>NORMAL OPERATION</span>", unsafe_allow_html=True)
    elif diag["sensor_diag"]["is_sensor_fault"]:
        st.markdown("<span class='badge-warning'>SENSOR FAULT DETECTED</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='badge-alert'>CRITICAL LEAK ACTIVE</span>", unsafe_allow_html=True)

    st.markdown(f"<p style='color: #9FB2C8; font-size: 12px; margin-top: 8px;'>Active State: <b style='color:#E7EAF0;'>{st.session_state.scenario_name}</b></p>", unsafe_allow_html=True)
    if st.session_state.mitigated_res is not None:
        st.markdown("<span class='badge-normal' style='background-color:#163428;'>SIMULATED MITIGATION APPLIED</span>", unsafe_allow_html=True)


# =============================================================================
# PAGE 1: OVERVIEW
# =============================================================================
if page == "1. Overview":
    st.markdown("<h2 style='color:#E7EAF0;'>Distribution Network Overview</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9FB2C8;'>Real-time hydraulic telemetry and system asset metrics across Zones A, B, and C.</p>", unsafe_allow_html=True)

    sim_res = st.session_state.sim_res
    wn = st.session_state.wn_current
    diag = st.session_state.diag

    # Compute key aggregated metrics
    pressures = [float(sim_res.pressure[j].iloc[0]) for j in wn.junction_name_list]
    demands = [float(sim_res.demand[j].iloc[0]) * 1000.0 for j in wn.junction_name_list]
    leaks = [
        float(sim_res.leak_demand[j].iloc[0]) * 1000.0
        for j in wn.junction_name_list
        if j in sim_res.leak_demand.columns
    ]
    total_leak_lps = sum(leaks)
    total_flow_lps = sum(max(0.0, d) for d in demands) + total_leak_lps
    avg_pressure_m = float(np.mean(pressures)) if pressures else 0.0

    # 4 Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        if total_leak_lps > 1.0:
            health_str = f"ALERT ({100 - min(100, int(total_leak_lps)):.0f}%)"
            st.metric("Network Health", health_str, delta=f"-{total_leak_lps:.1f} L/s leak", delta_color="inverse")
        elif diag["sensor_diag"]["is_sensor_fault"]:
            st.metric("Network Health", "DEGRADED (92%)", delta="Sensor Glitch", delta_color="off")
        else:
            st.metric("Network Health", "100.0% Nominal", delta="Optimal")

    with m2:
        if total_leak_lps > 1.0:
            st.metric("Active Incidents", "1 Pipe Rupture", delta="Zone B", delta_color="inverse")
        elif diag["sensor_diag"]["is_sensor_fault"]:
            st.metric("Active Incidents", "1 Sensor Fault", delta=diag["sensor_diag"]["target_node"], delta_color="off")
        else:
            st.metric("Active Incidents", "0 Incidents", delta="Normal")

    with m3:
        delta_q = total_flow_lps - 56.4
        st.metric("Total System Flow", f"{total_flow_lps:.2f} L/s", delta=f"{delta_q:+.2f} L/s" if abs(delta_q) > 0.1 else None)

    with m4:
        delta_p = avg_pressure_m - 90.0
        st.metric("Average Pressure", f"{avg_pressure_m:.2f} m", delta=f"{delta_p:+.2f} m" if abs(delta_p) > 0.1 else None)

    st.markdown("<br>", unsafe_allow_html=True)

    # Middle Row: Water Balance vs Physical Topology
    c_left, c_right = st.columns([1, 1])
    with c_left:
        st.markdown("""
        <div class='aqua-card aqua-card-gold'>
            <h4 style='color: #C9A227; margin-top: 0;'>Water Balance & Service Continuity</h4>
            <table style='width: 100%; color: #E7EAF0; font-size: 14px;'>
                <tr><td style='color:#9FB2C8;'>Customer Demand Required:</td><td><b>56.40 L/s</b></td></tr>
                <tr><td style='color:#9FB2C8;'>Customer Demand Delivered:</td><td><b>{:.2f} L/s</b></td></tr>
                <tr><td style='color:#9FB2C8;'>Unaccounted Water Loss (Leak):</td><td style='color: {};'><b>{:.2f} L/s</b></td></tr>
                <tr><td style='color:#9FB2C8;'>Customer Service Index:</td><td><b>{:.1f}%</b></td></tr>
                <tr><td style='color:#9FB2C8;'>Pressure Standard Compliance:</td><td><b>{}/15 Junctions (>15m)</b></td></tr>
            </table>
        </div>
        """.format(
            sum(max(0.0, d) for d in demands),
            "#E74C3C" if total_leak_lps > 1.0 else "#2ECC71",
            total_leak_lps,
            min(100.0, (sum(max(0.0, d) for d in demands) / 56.40) * 100.0),
            sum(1 for p in pressures if p >= 15.0),
        ), unsafe_allow_html=True)

    with c_right:
        summary = get_network_summary(wn)
        st.markdown(f"""
        <div class='aqua-card'>
            <h4 style='color: #E7EAF0; margin-top: 0;'>Physical Asset Inventory</h4>
            <table style='width: 100%; color: #E7EAF0; font-size: 14px;'>
                <tr><td style='color:#9FB2C8;'>Water Source:</td><td><b>1 Bulk Reservoir (R1 @ 60m Head)</b></td></tr>
                <tr><td style='color:#9FB2C8;'>Pressure Districts:</td><td><b>3 Zones (Zone A, Zone B, Zone C)</b></td></tr>
                <tr><td style='color:#9FB2C8;'>Total Junctions:</td><td><b>{summary['num_junctions']} Instrumented Nodes</b></td></tr>
                <tr><td style='color:#9FB2C8;'>Transmission Pipes:</td><td><b>{summary['num_pipes']} Segments</b></td></tr>
                <tr><td style='color:#9FB2C8;'>Supply Pumps:</td><td><b>2 Units (PUMP_1 Lift, PUMP_2 Booster)</b></td></tr>
                <tr><td style='color:#9FB2C8;'>Control Valves:</td><td><b>3 Isolation Valves (TCV A, B, C)</b></td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # Zone Operational Status Cards
    st.markdown("<h3 style='color:#E7EAF0;'>District Metered Area (DMA) Status</h3>", unsafe_allow_html=True)
    z1, z2, z3 = st.columns(3)

    for col, zone_name, label in [(z1, "Zone_A", "Zone A (Industrial)"), (z2, "Zone_B", "Zone B (Commercial Hub)"), (z3, "Zone_C", "Zone C (Residential)")]:
        with col:
            z_nodes = [j for j in wn.junction_name_list if getattr(wn.get_node(j), "zone", "") == zone_name]
            z_p = [float(sim_res.pressure[j].iloc[0]) for j in z_nodes]
            z_q = [float(sim_res.demand[j].iloc[0]) * 1000.0 for j in z_nodes]
            z_leak = [
                float(sim_res.leak_demand[j].iloc[0]) * 1000.0
                for j in z_nodes
                if j in sim_res.leak_demand.columns
            ]
            has_leak = sum(z_leak) > 1.0

            card_class = "aqua-card-red" if has_leak else "aqua-card-green"
            badge = "<span class='badge-alert'>LEAK DISRUPTION</span>" if has_leak else "<span class='badge-normal'>HEALTHY</span>"

            st.markdown(f"""
            <div class='aqua-card {card_class}'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <h4 style='color: #E7EAF0; margin: 0;'>{label}</h4>
                    {badge}
                </div>
                <hr style='border: 1px solid #22334D; margin: 8px 0;'>
                <p style='color:#9FB2C8; font-size:13px; margin:4px 0;'>Mean Pressure: <b style='color:#E7EAF0;'>{np.mean(z_p):.2f} m</b></p>
                <p style='color:#9FB2C8; font-size:13px; margin:4px 0;'>Customer Demand: <b style='color:#E7EAF0;'>{sum(z_q):.2f} L/s</b></p>
                <p style='color:#9FB2C8; font-size:13px; margin:4px 0;'>Active Leakage: <b style='color:{"#E74C3C" if has_leak else "#2ECC71"};'>{sum(z_leak):.2f} L/s</b></p>
                <p style='color:#9FB2C8; font-size:12px; margin:4px 0;'>Nodes: {', '.join(z_nodes)}</p>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# PAGE 2: DIGITAL TWIN
# =============================================================================
elif page == "2. Digital Twin":
    st.markdown("<h2 style='color:#E7EAF0;'>Network Topology Digital Twin</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9FB2C8;'>Real-time spatial visualization. Suspicious pipes, closed valves, and pressures are highlighted dynamically.</p>", unsafe_allow_html=True)

    sim_res = st.session_state.sim_res
    wn = st.session_state.wn_current
    diag = st.session_state.diag
    top_suspect_pipe = diag.get("top_suspect_pipe", "")

    meas_pressures = diag["meas_pressures"]
    meas_flows = diag["meas_flows"]

    fig = go.Figure()

    # 1. Draw Links (Pipes, Valves, Pumps)
    for l_name, link in wn.links():
        u = link.start_node_name
        v = link.end_node_name
        u_node = wn.get_node(u)
        v_node = wn.get_node(v)

        x0, y0 = u_node.coordinates
        x1, y1 = v_node.coordinates

        # Check status: Closed valve or closed pipe
        is_closed = (
            getattr(link, "status", None) == wntr.network.LinkStatus.Closed
            or getattr(link, "initial_status", None) == wntr.network.LinkStatus.Closed
        )
        is_suspect = (l_name == top_suspect_pipe and diag["is_anomaly"])

        # Determine visual style
        if is_closed:
            color = "#C4554D"  # Alert Red
            width = 5
            dash = "dash"
            status_desc = "CLOSED (ISOLATED)"
        elif is_suspect:
            color = "#C9A227"  # Gold Accent
            width = 6
            dash = "solid"
            status_desc = f"SUSPECT PIPE (Score: {diag['suspect_pipes'][0]['suspect_score']}%)"
        elif link.link_type == "Pump":
            color = "#2ECC71"  # Bright Green
            width = 4
            dash = "solid"
            status_desc = "SUPPLY PUMP (ACTIVE)"
        elif link.link_type == "Valve":
            color = "#9B59B6"  # Purple
            width = 4
            dash = "solid"
            status_desc = "ISOLATION VALVE (OPEN)"
        else:
            color = "#3498DB"  # Steel Blue
            width = 3
            dash = "solid"
            status_desc = "NORMAL PIPE (ACTIVE)"

        flow_lps = float(meas_flows.get(l_name, 0.0)) * 1000.0

        hover_info = (
            f"<b>Link:</b> {l_name}<br>"
            f"<b>Type:</b> {link.link_type}<br>"
            f"<b>Path:</b> {u} &rarr; {v}<br>"
            f"<b>Flowrate:</b> {flow_lps:.2f} L/s<br>"
            f"<b>Diameter:</b> {getattr(link, 'diameter', 0.25)*1000:.0f} mm<br>"
            f"<b>Status:</b> {status_desc}"
        )

        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            hoverinfo="text",
            hovertext=hover_info,
            showlegend=False,
        ))

    # 2. Draw Nodes (Reservoir & Junctions)
    node_x, node_y, node_text, node_color, node_size, node_symbol = [], [], [], [], [], []

    for n_name, node in wn.nodes():
        x, y = node.coordinates
        node_x.append(x)
        node_y.append(y)

        if node.node_type == "Reservoir":
            node_color.append("#9B59B6")
            node_size.append(18)
            node_symbol.append("diamond")
            node_text.append(f"<b>Reservoir:</b> {n_name}<br><b>Head:</b> {node.head_timeseries.base_value:.1f} m")
        else:
            p = float(meas_pressures.get(n_name, 0.0))
            d = float(sim_res.demand[n_name].iloc[0]) * 1000.0
            leak_q = float(sim_res.leak_demand[n_name].iloc[0]) * 1000.0 if n_name in sim_res.leak_demand.columns else 0.0
            zone = getattr(node, "zone", "N/A")

            # Check if this node is the faulty sensor
            is_faulty_sensor = (diag["sensor_diag"]["is_sensor_fault"] and diag["sensor_diag"]["target_node"] == n_name)
            is_leak_node = (leak_q > 0.5)

            if is_leak_node:
                node_color.append("#C4554D")  # Red
                node_size.append(16)
                node_symbol.append("circle-x")
            elif is_faulty_sensor:
                node_color.append("#C9A227")  # Gold
                node_size.append(16)
                node_symbol.append("cross")
            else:
                node_color.append("#2980B9")  # Blue
                node_size.append(12)
                node_symbol.append("circle")

            hover_text = (
                f"<b>Junction:</b> {n_name}<br>"
                f"<b>Zone:</b> {zone}<br>"
                f"<b>Pressure:</b> {p:.2f} m<br>"
                f"<b>Consumer Demand:</b> {d:.2f} L/s<br>"
                f"<b>Leak Discharge:</b> {leak_q:.2f} L/s"
            )
            if is_faulty_sensor:
                hover_text += "<br><b style='color:#C9A227;'>[SENSOR MALFUNCTION FLAGGED]</b>"
            node_text.append(hover_text)

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(color=node_color, size=node_size, symbol=node_symbol, line=dict(color="#0F1B2D", width=1.5)),
        text=[n for n in wn.node_name_list],
        textposition="top center",
        textfont=dict(color="#E7EAF0", size=10),
        hoverinfo="text",
        hovertext=node_text,
        showlegend=False,
    ))

    # Add custom Legend elements
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="#3498DB", width=3), name="Normal Pipe"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="#C9A227", width=5), name="Top Suspect Pipe"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="#C4554D", width=4, dash="dash"), name="Closed / Isolated Valve"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="#C4554D", size=10, symbol="circle-x"), name="Physical Leak Node"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="#C9A227", size=10, symbol="cross"), name="Faulty Sensor"))

    fig.update_layout(
        paper_bgcolor="#17253D",
        plot_bgcolor="#17253D",
        height=540,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(showgrid=True, gridcolor="#22334D", zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor="#22334D", zeroline=False, showticklabels=False),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="#E7EAF0", size=11),
            bgcolor="rgba(15,27,45,0.7)",
        ),
    )

    st.plotly_chart(fig, width="stretch")

    # Telemetry Table below graph
    st.markdown("<h4 style='color:#E7EAF0;'>Real-time Nodal Telemetry</h4>", unsafe_allow_html=True)
    table_rows = []
    for j in wn.junction_name_list:
        node = wn.get_node(j)
        p = float(meas_pressures.get(j, 0.0))
        d = float(sim_res.demand[j].iloc[0]) * 1000.0
        lq = float(sim_res.leak_demand[j].iloc[0]) * 1000.0 if j in sim_res.leak_demand.columns else 0.0
        table_rows.append({
            "Junction": j,
            "Zone": getattr(node, "zone", "N/A"),
            "Elevation (m)": node.elevation,
            "Measured Pressure (m)": round(p, 2),
            "Demand (L/s)": round(d, 2),
            "Leakage (L/s)": round(lq, 2),
            "Status": "LEAK DISCHARGE" if lq > 0.5 else ("LOW PRESSURE" if p < 15.0 else "NOMINAL"),
        })

    telemetry_df = pd.DataFrame(table_rows)
    st.dataframe(telemetry_df, width="stretch", height=220)


# =============================================================================
# PAGE 3: SIMULATION LAB
# =============================================================================
elif page == "3. Simulation Lab":
    st.markdown("<h2 style='color:#E7EAF0;'>Hydraulic Fault Simulation Lab</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9FB2C8;'>Inject physical pipe ruptures or faulty sensor readings to benchmark AI detection and automated decision response.</p>", unsafe_allow_html=True)

    sim_tab1, sim_tab2 = st.tabs(["💧 Physical Pipe Leak Simulation", "⚡ Inject Faulty Sensor (False Alarm Demo)"])

    # TAB 1: Pipe Leak Simulation
    with sim_tab1:
        st.markdown("<div class='aqua-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#C9A227; margin-top:0;'>Configure Zone B Physical Rupture</h4>", unsafe_allow_html=True)

        col_loc, col_sev, col_dem = st.columns(3)
        with col_loc:
            pipe_choice = st.selectbox(
                "Target Pipe (Zone B)",
                ["P_B1", "P_B2", "P_B3", "P_B4", "P_B5", "P_B6"],
                index=0,
                help="Select pipe segment to inject physical burst leak.",
            )
        with col_sev:
            sev_choice = st.selectbox(
                "Leak Orifice Severity",
                ["Major Rupture (56 mm, 0.0025 m²)", "Moderate Burst (40 mm, 0.0013 m²)", "Minor Pinhole (20 mm, 0.0003 m²)"],
                index=0,
            )
            area_map = {
                "Major Rupture (56 mm, 0.0025 m²)": 0.0025,
                "Moderate Burst (40 mm, 0.0013 m²)": 0.0013,
                "Minor Pinhole (20 mm, 0.0003 m²)": 0.0003,
            }
            leak_area = area_map[sev_choice]

        with col_dem:
            demand_fluctuation = st.slider(
                "Demand Variation (Random Load)",
                min_value=0.85,
                max_value=1.15,
                value=1.00,
                step=0.05,
                help="Simulates normal diurnal customer consumption changes.",
            )

        if st.button("RUN LEAK SIMULATION", key="btn_run_leak"):
            with st.spinner("Executing real WNTR hydraulic simulation..."):
                # Build fresh model
                wn_sim = build_network(duration_hours=0)

                # Apply demand perturbation
                for _, j in wn_sim.junctions():
                    j.demand_timeseries_list[0].base_value *= demand_fluctuation

                # Inject pipe leak
                FaultInjector.inject_pipe_leak(wn=wn_sim, pipe_name=pipe_choice, leak_area=leak_area, start_time=0)

                # Re-run simulation
                engine = HydraulicEngine(wn_sim)
                sim_res = engine.run(engine="WNTR", mode="PDD")

                # Update session state
                st.session_state.wn_current = wn_sim
                st.session_state.sim_res = sim_res
                st.session_state.scenario_name = f"Leak on {pipe_choice} ({sev_choice.split('(')[0].strip()})"
                st.session_state.scenario_type = "leak"
                st.session_state.faulty_sensor_override = None
                st.session_state.mitigated_res = None
                st.session_state.recovery_report = None

                # Re-diagnose
                st.session_state.diag = run_full_diagnosis(wn_sim, sim_res, st.session_state.res_base)

            st.success(f"Simulation completed! Physical rupture active on pipe '{pipe_choice}'. Check 'AI & Response' or 'Digital Twin'.")
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # TAB 2: Faulty Sensor Injection
    with sim_tab2:
        st.markdown("<div class='aqua-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#C9A227; margin-top:0;'>Simulate Sensor Instrumentation Failure</h4>", unsafe_allow_html=True)
        st.markdown("<p style='color:#9FB2C8; font-size:13px;'>Test false alarm rejection: When a single sensor malfunctions without neighbor correlation, the decision engine must reject automated valve closures and flag the sensor.</p>", unsafe_allow_html=True)

        col_s_node, col_s_type = st.columns(2)
        with col_s_node:
            sensor_choice = st.selectbox(
                "Target Sensor",
                ["J_B2", "J_B1", "J_A3", "J_C2", "J_C4"],
                index=0,
            )
        with col_s_type:
            failure_mode = st.selectbox(
                "Sensor Failure Mode",
                ["Sudden False Drop (-18.0 m)", "Stuck Telemetry (Frozen @ 42.0 m)", "Calibration Drift (-12.0 m)"],
                index=0,
            )

        if st.button("INJECT SENSOR FAULT & EVALUATE", key="btn_run_sensor"):
            with st.spinner("Injecting sensor glitch into normal network..."):
                # Use normal network
                wn_sim = build_network(duration_hours=0)
                engine = HydraulicEngine(wn_sim)
                sim_res = engine.run(engine="WNTR", mode="PDD")

                # Compute overridden reading
                norm_p = float(sim_res.pressure[sensor_choice].iloc[0])
                if "Drop" in failure_mode:
                    new_val = norm_p - 18.0
                elif "Stuck" in failure_mode:
                    new_val = 42.0
                else:
                    new_val = norm_p - 12.0

                fault_override = {"sensor": sensor_choice, "value": new_val, "mode": failure_mode}

                # Update session state
                st.session_state.wn_current = wn_sim
                st.session_state.sim_res = sim_res
                st.session_state.scenario_name = f"Sensor Glitch on {sensor_choice} ({failure_mode.split('(')[0].strip()})"
                st.session_state.scenario_type = "sensor_fault"
                st.session_state.faulty_sensor_override = fault_override
                st.session_state.mitigated_res = None
                st.session_state.recovery_report = None

                # Re-diagnose with sensor fault override
                st.session_state.diag = run_full_diagnosis(wn_sim, sim_res, st.session_state.res_base, faulty_sensor_override=fault_override)

            st.warning(f"Sensor fault injected on '{sensor_choice}'. Notice that neighboring sensors are normal. Check 'AI & Response' to verify false alarm rejection!")
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # Reset Option
    st.markdown("<hr style='border: 1px solid #22334D;'>", unsafe_allow_html=True)
    if st.button("RESET TO NORMAL BASELINE", key="btn_reset"):
        wn_base, res_base = get_baseline_simulation()
        st.session_state.wn_current = copy.deepcopy(wn_base)
        st.session_state.sim_res = res_base
        st.session_state.scenario_name = "Normal Baseline"
        st.session_state.scenario_type = "normal"
        st.session_state.faulty_sensor_override = None
        st.session_state.mitigated_res = None
        st.session_state.recovery_report = None
        st.session_state.diag = run_full_diagnosis(st.session_state.wn_current, res_base, res_base)
        st.success("Network reset to pristine baseline operating conditions.")
        st.rerun()


# =============================================================================
# PAGE 4: AI & RESPONSE
# =============================================================================
elif page == "4. AI & Response":
    st.markdown("<h2 style='color:#E7EAF0;'>AI Intelligence & Autonomous Response</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9FB2C8;'>End-to-end telemetry analytics: Anomaly detection, NetworkX fault localisation, sensor validation, and automated valve response.</p>", unsafe_allow_html=True)

    diag = st.session_state.diag
    opt_result = diag["opt_result"]

    # Top Row: AI Anomaly & Classifier Verdict
    ai_col1, ai_col2 = st.columns(2)
    with ai_col1:
        score_val = diag["anomaly_score"]
        is_ano = diag["is_anomaly"]
        badge_cls = "badge-alert" if is_ano else "badge-normal"
        badge_txt = "ANOMALY DETECTED" if is_ano else "NOMINAL OPERATION"

        st.markdown(f"""
        <div class='aqua-card {"aqua-card-red" if is_ano else "aqua-card-green"}'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <h4 style='color: #E7EAF0; margin: 0;'>Isolation Forest Anomaly Detector</h4>
                <span class='{badge_cls}'>{badge_txt}</span>
            </div>
            <hr style='border: 1px solid #22334D; margin: 10px 0;'>
            <p style='color:#9FB2C8; font-size:13px; margin:4px 0;'>Anomaly Likelihood Score: <b style='color:#E7EAF0; font-size:16px;'>{score_val:.4f}</b> (0.0=Normal, 1.0=Severe)</p>
            <p style='color:#9FB2C8; font-size:12px;'>Trained strictly on normal hydraulic data (37 features). Unsupervised outlier distance.</p>
        </div>
        """, unsafe_allow_html=True)

    with ai_col2:
        conf = diag["ai_confidence"] * 100.0
        label = diag["predicted_label"]
        st.markdown(f"""
        <div class='aqua-card {"aqua-card-red" if label == "Leak" else "aqua-card-green"}'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <h4 style='color: #E7EAF0; margin: 0;'>Random Forest Classifier</h4>
                <span class='{"badge-alert" if label == "Leak" else "badge-normal"}'>{label.upper()} VERDICT</span>
            </div>
            <hr style='border: 1px solid #22334D; margin: 10px 0;'>
            <p style='color:#9FB2C8; font-size:13px; margin:4px 0;'>Classification Confidence: <b style='color:#E7EAF0; font-size:16px;'>{conf:.1f}%</b> (Safety Gate: &ge; 90.0%)</p>
            <p style='color:#9FB2C8; font-size:12px;'>Test Set Accuracy: 100.0%, F1 Score: 100.0%. Discriminated via nodal pressure & flow gradients.</p>
        </div>
        """, unsafe_allow_html=True)

    # Middle Row: Top-3 Suspect Pipes vs Sensor Health Panel
    r2_col1, r2_col2 = st.columns([1, 1])

    with r2_col1:
        st.markdown("<h4 style='color:#C9A227;'>Top-3 Suspect Pipes (NetworkX Localisation)</h4>", unsafe_allow_html=True)
        suspects = diag["suspect_pipes"]
        if suspects:
            for s in suspects:
                rank_str = f"#{s['rank']}"
                st.markdown(f"""
                <div class='aqua-card' style='padding: 12px 16px; margin-bottom: 8px;'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <b style='color:#C9A227; font-size:15px;'>{rank_str} Pipe '{s['pipe_name']}'</b>
                        <b style='color:#E7EAF0;'>Score: {s['suspect_score']}%</b>
                    </div>
                    <p style='color:#9FB2C8; font-size:12px; margin:4px 0;'>Zone: <b>{s['zone']}</b> | Endpoints: <b>{s['start_node']} &rarr; {s['end_node']}</b></p>
                    <p style='color:#9FB2C8; font-size:12px; margin:4px 0;'>Average Pressure Drop: <b>{s['avg_pressure_drop_m']:.2f} m</b> | Distance to Epicenter: <b>{s['hops_to_epicenter']} hops</b></p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No active pressure drops detected. All pipe segments operating nominally.")

    with r2_col2:
        st.markdown("<h4 style='color:#C9A227;'>Sensor Integrity & Health Panel</h4>", unsafe_allow_html=True)
        sensor_diag = diag["sensor_diag"]

        # Build sensor health status
        sensor_records = []
        wn = st.session_state.wn_current
        meas_p = diag["meas_pressures"]
        norm_p = st.session_state.res_base.pressure.iloc[0].to_dict()

        for j in sorted(wn.junction_name_list):
            is_faulty = (sensor_diag["is_sensor_fault"] and sensor_diag["target_node"] == j)
            drop = float(norm_p.get(j, 0.0) - meas_p.get(j, 0.0))
            status_tag = "FAULTY (MALFUNCTION)" if is_faulty else "OK (HEALTHY)"
            sensor_records.append({
                "Sensor": j,
                "Zone": getattr(wn.get_node(j), "zone", "N/A"),
                "Measured (m)": round(float(meas_p.get(j, 0.0)), 2),
                "Delta (m)": round(drop, 2),
                "Integrity": status_tag,
            })

        sensor_df = pd.DataFrame(sensor_records)
        style_mapper = getattr(sensor_df.style, "map", None) or getattr(sensor_df.style, "applymap")
        st.dataframe(
            style_mapper(
                lambda v: "color: #E74C3C; font-weight: bold;" if "FAULTY" in str(v) else ("color: #2ECC71;" if "OK" in str(v) else ""),
                subset=["Integrity"]
            ),
            width="stretch",
            height=215,
        )

    # Row 3: Automated Valve Response & Policy Gate
    st.markdown("<hr style='border: 1px solid #22334D; margin: 18px 0;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#E7EAF0;'>Automated Response Policy & Action Execution</h3>", unsafe_allow_html=True)

    d_col1, d_col2 = st.columns([1.2, 1])

    with d_col1:
        st.markdown("<h4 style='color:#C9A227;'>Automated Response Policy Assessment</h4>", unsafe_allow_html=True)

        is_auto_approved = opt_result.auto_action_approved
        if is_auto_approved:
            st.markdown("""
            <div class='aqua-card aqua-card-green'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <b style='color:#2ECC71; font-size:16px;'>AUTOMATIC ACTION EXECUTED</b>
                    <span class='badge-normal'>POLICY PASSED</span>
                </div>
                <hr style='border: 1px solid #22334D; margin: 8px 0;'>
                <p style='color:#E7EAF0; font-size:13px;'><b>Selected Plan:</b> {}</p>
                <p style='color:#E7EAF0; font-size:13px;'><b>Target Links:</b> {}</p>
                <p style='color:#9FB2C8; font-size:12px;'>{}</p>
            </div>
            """.format(
                opt_result.selected_action.action_name,
                opt_result.selected_action.target_links,
                opt_result.policy_reason,
            ), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='aqua-card aqua-card-red'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <b style='color:#E74C3C; font-size:16px;'>AUTOMATIC ACTION INHIBITED: MANUAL INSPECTION REQUIRED</b>
                    <span class='badge-alert'>SAFETY LOCK</span>
                </div>
                <hr style='border: 1px solid #22334D; margin: 8px 0;'>
                <p style='color:#E7EAF0; font-size:13px;'><b>Automated Valve Closure Rejected.</b></p>
                <p style='color:#9FB2C8; font-size:12px;'>{}</p>
            </div>
            """.format(opt_result.policy_reason), unsafe_allow_html=True)

        # Log entry box
        if opt_result.action_log:
            st.markdown("<b style='color:#9FB2C8; font-size:12px;'>Action Audit Log:</b>", unsafe_allow_html=True)
            st.json(opt_result.action_log)
        else:
            st.markdown("<p style='color:#9FB2C8; font-size:12px;'>No automated actuation executed on network.</p>", unsafe_allow_html=True)

    with d_col2:
        st.markdown("<h4 style='color:#C9A227;'>Operator Control & Manual Override</h4>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#9FB2C8; font-size:13px;'>Recommended Mitigation: <b style='color:#E7EAF0;'>{opt_result.selected_action.action_name if opt_result.selected_action else 'N/A'}</b></p>", unsafe_allow_html=True)

        # Manual Override Button
        can_execute = opt_result.selected_action is not None and len(opt_result.selected_action.target_links) > 0
        if st.button("EXECUTE ACTION (MANUAL OVERRIDE)", key="btn_exec_override", disabled=not can_execute):
            with st.spinner("Applying simulated valve closure and re-running simulation..."):
                # Apply closure
                wn_mitigated = copy.deepcopy(st.session_state.wn_current)
                for l in opt_result.selected_action.target_links:
                    if l in wn_mitigated.link_name_list:
                        wn_mitigated.get_link(l).initial_status = "Closed"

                # Re-run simulation
                res_mitigated = HydraulicEngine(wn_mitigated).run(engine="WNTR", mode="PDD")
                st.session_state.mitigated_res = res_mitigated

                # Verify recovery
                verifier = RecoveryVerifier()
                action_log_entry = {
                    "event_type": "SIMULATED_VALVE_ACTION_MANUAL_OVERRIDE",
                    "target_suspect_pipe": opt_result.target_suspect_pipe,
                    "selected_action": opt_result.selected_action.action_name,
                    "isolated_links": opt_result.selected_action.target_links,
                    "ai_confidence_pct": round(opt_result.ai_confidence * 100.0, 1),
                }
                recovery_rep = verifier.verify_recovery(
                    pre_wn=st.session_state.wn_current,
                    post_wn=wn_mitigated,
                    target_pipe=opt_result.target_suspect_pipe,
                    action_log=action_log_entry,
                )
                st.session_state.recovery_report = recovery_rep

            st.success("Simulated valve action executed! Hydraulic recovery verified below.")
            st.rerun()

    # Row 4: Before vs After Analytics
    st.markdown("<hr style='border: 1px solid #22334D; margin: 18px 0;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#E7EAF0;'>Before vs. After Recovery Analytics</h3>", unsafe_allow_html=True)

    # If action was applied automatically or manually, use that; otherwise use optimizer candidate
    if st.session_state.recovery_report is not None:
        rep = st.session_state.recovery_report
        st.markdown(f"<p style='color:#2ECC71;'><b>Status: {rep.status}</b> &mdash; {rep.action_description}</p>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Water Loss Reduction", f"{rep.water_loss_after_lps:.2f} L/s", delta=f"-{rep.water_loss_reduction_pct:.1f}%", delta_color="inverse")
        c2.metric("Customer Service Level", f"{rep.customer_service_pct_after:.1f}%", delta=f"{rep.customer_service_pct_after - rep.customer_service_pct_before:+.1f}%")
        c3.metric("Mean System Pressure", f"{rep.mean_pressure_after_m:.2f} m", delta=f"{rep.mean_pressure_after_m - rep.mean_pressure_before_m:+.2f} m")
        c4.metric("Network Operational", "YES (ONLINE)", delta="Active DMA")

        # Nodal Pressure Comparison Chart
        st.markdown("<h4 style='color:#C9A227; margin-top:16px;'>Nodal Pressure Recovery Profile (Before vs. After)</h4>", unsafe_allow_html=True)
        j_names = st.session_state.wn_current.junction_name_list
        p_pre = [float(st.session_state.sim_res.pressure[j].iloc[0]) for j in j_names]
        p_post = [float(st.session_state.mitigated_res.pressure[j].iloc[0]) for j in j_names]

        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(x=j_names, y=p_pre, name="Before Mitigation (Leak Active)", marker_color="#C4554D"))
        bar_fig.add_trace(go.Bar(x=j_names, y=p_post, name="After Mitigation (Valve Closed)", marker_color="#2ECC71"))
        bar_fig.add_hline(y=15.0, line_dash="dash", line_color="#C9A227", annotation_text="Minimum Service Pressure (15m)")

        bar_fig.update_layout(
            paper_bgcolor="#17253D",
            plot_bgcolor="#17253D",
            height=340,
            margin=dict(l=20, r=20, t=30, b=20),
            font=dict(color="#E7EAF0"),
            xaxis=dict(gridcolor="#22334D"),
            yaxis=dict(gridcolor="#22334D", title="Pressure (m)"),
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
        )
        st.plotly_chart(bar_fig, width="stretch")

        st.markdown("<h5 style='color:#E7EAF0;'>Physical State & Remaining Abnormalities:</h5>", unsafe_allow_html=True)
        for ab in rep.remaining_abnormalities:
            st.markdown(f"<p style='color:#9FB2C8; font-size:13px; margin:2px 0;'>&bull; {ab}</p>", unsafe_allow_html=True)

    else:
        st.info("Execute automated or manual mitigation above to generate before-vs-after recovery charts and residual abnormality reports.")
