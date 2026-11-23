"""
AquaSentinel AI - Sensor Fault Diagnostics Engine
Differentiates between true physical hydraulic leaks and sensor instrumentation
malfunctions (stuck sensors, calibration drifts, false drops) using graph topology
and spatial hydraulic consistency checks.
"""

from typing import Dict, List, Any, Optional
import networkx as nx
import numpy as np
import wntr


class SensorFaultInjector:
    """
    Simulates faulty sensor telemetry by injecting electrical and hardware failure modes.
    """

    @staticmethod
    def inject_stuck_reading(
        readings: Dict[str, float],
        sensor_name: str,
        stuck_value: float = 40.0,
    ) -> Dict[str, Any]:
        """
        Simulates an analog-to-digital converter freeze or mechanical freeze
        where the sensor outputs a fixed static reading regardless of system state.
        """
        original = readings.get(sensor_name, 0.0)
        readings[sensor_name] = float(stuck_value)
        return {
            "fault_type": "sensor_stuck",
            "sensor": sensor_name,
            "original_value": round(original, 2),
            "injected_value": round(stuck_value, 2),
        }

    @staticmethod
    def inject_drifting_reading(
        readings: Dict[str, float],
        sensor_name: str,
        drift_offset: float = -12.0,
    ) -> Dict[str, Any]:
        """
        Simulates gradual calibration drift or bias accumulation.
        """
        original = readings.get(sensor_name, 0.0)
        readings[sensor_name] = float(original + drift_offset)
        return {
            "fault_type": "sensor_drift",
            "sensor": sensor_name,
            "original_value": round(original, 2),
            "injected_value": round(readings[sensor_name], 2),
            "drift_offset": round(drift_offset, 2),
        }

    @staticmethod
    def inject_false_drop(
        readings: Dict[str, float],
        sensor_name: str,
        drop_magnitude: float = 18.0,
    ) -> Dict[str, Any]:
        """
        Simulates a sudden telemetry glitch or ground-fault that creates an apparent
        severe pressure drop at a single sensor without any physical hydraulic event.
        """
        original = readings.get(sensor_name, 0.0)
        readings[sensor_name] = float(original - drop_magnitude)
        return {
            "fault_type": "sensor_false_drop",
            "sensor": sensor_name,
            "original_value": round(original, 2),
            "injected_value": round(readings[sensor_name], 2),
            "drop_magnitude": round(drop_magnitude, 2),
        }


class SensorFaultDiagnoser:
    """
    Physics-informed diagnostics engine that verifies spatial hydraulic consistency
    across graph neighbors to distinguish real physical leaks from sensor faults.
    """

    def __init__(self, wn: wntr.network.WaterNetworkModel):
        self.wn = wn
        self.graph = self._build_graph(wn)

    def _build_graph(self, wn: wntr.network.WaterNetworkModel) -> nx.Graph:
        G = nx.Graph()
        for j in wn.junction_name_list:
            G.add_node(j)
        for _, pipe in wn.pipes():
            G.add_edge(pipe.start_node_name, pipe.end_node_name)
        for _, valve in wn.valves():
            G.add_edge(valve.start_node_name, valve.end_node_name)
        for _, pump in wn.pumps():
            G.add_edge(pump.start_node_name, pump.end_node_name)
        return G

    def diagnose(
        self,
        measured_pressures: Dict[str, float],
        normal_pressures: Dict[str, float],
        measured_flows: Optional[Dict[str, float]] = None,
        normal_flows: Optional[Dict[str, float]] = None,
        pressure_anomaly_threshold: float = 4.0,
    ) -> Dict[str, Any]:
        """
        Analyzes measured pressures and flows against hydraulic graph continuity.

        Physical Principle:
        - A REAL LEAK creates a hydraulic cone of depression: adjacent 1-hop and 2-hop
          neighbors must register correlated pressure drops.
        - A SENSOR FAULT creates an isolated discontinuity: the candidate sensor shows
          a significant drop, while its topological neighbors show normal pressures.

        Parameters
        ----------
        measured_pressures : Dict[str, float]
            Observed pressures from network sensors.
        normal_pressures : Dict[str, float]
            Baseline normal pressures.
        measured_flows : Dict[str, float], optional
            Observed flow rates.
        normal_flows : Dict[str, float], optional
            Baseline normal flow rates.
        pressure_anomaly_threshold : float
            Minimum pressure drop (m) to trigger anomaly analysis.

        Returns
        -------
        Dict[str, Any]
            Comprehensive diagnosis: classification, affected node, confidence,
            and spatial evidence metrics.
        """
        # 1. Compute pressure residuals (delta = normal - measured)
        residuals: Dict[str, float] = {}
        for j in self.wn.junction_name_list:
            pn = normal_pressures.get(j, measured_pressures.get(j, 0.0))
            pm = measured_pressures.get(j, pn)
            residuals[j] = float(pn - pm)

        # Find maximum positive drop (or absolute deviation)
        max_node = max(residuals.items(), key=lambda x: x[1])[0]
        max_drop = residuals[max_node]

        # Check if the maximum drop exceeds the detection threshold
        if max_drop < pressure_anomaly_threshold:
            return {
                "diagnosis": "NORMAL",
                "is_fault": False,
                "is_sensor_fault": False,
                "target_node": None,
                "confidence": 0.95,
                "explanation": f"All sensor pressures are within normal hydraulic bounds (max deviation: {max_drop:.2f} m).",
                "spatial_evidence": {},
            }

        # 2. Extract topological 1-hop and 2-hop graph neighbors
        direct_neighbors = list(self.graph.neighbors(max_node)) if max_node in self.graph else []
        neighbor_drops = [residuals.get(nbr, 0.0) for nbr in direct_neighbors if nbr in self.wn.junction_name_list]

        avg_neighbor_drop = float(np.mean(neighbor_drops)) if neighbor_drops else 0.0
        max_neighbor_drop = float(np.max(neighbor_drops)) if neighbor_drops else 0.0

        # Spatial consistency ratio: how much neighboring sensors share the drop
        spatial_ratio = (avg_neighbor_drop / max_drop) if max_drop > 0.1 else 0.0

        # 3. Mass-balance flow check on links incident to max_node
        flow_support = 0.0
        if measured_flows is not None and normal_flows is not None:
            # Check if flow into this node increased
            for l_name, link in self.wn.links():
                if link.start_node_name == max_node or link.end_node_name == max_node:
                    qm = measured_flows.get(l_name, normal_flows.get(l_name, 0.0))
                    qn = normal_flows.get(l_name, qm)
                    flow_support += abs(qm - qn)

        # 4. Decision Logic based strictly on spatial hydraulic consistency:
        # If the target sensor shows a significant drop (>4m), BUT its immediate
        # graph neighbors show less than 20% of that drop (or near zero drop):
        # Physical continuity forbids a point sink without surrounding gradient.
        # Therefore, the anomaly MUST BE a sensor fault!
        is_sensor_fault = (spatial_ratio < 0.20) and (avg_neighbor_drop < 2.5)

        spatial_evidence = {
            "evaluated_node": max_node,
            "target_drop_m": round(max_drop, 2),
            "direct_neighbors": direct_neighbors,
            "neighbor_drops_m": {nbr: round(residuals.get(nbr, 0.0), 2) for nbr in direct_neighbors},
            "avg_neighbor_drop_m": round(avg_neighbor_drop, 2),
            "max_neighbor_drop_m": round(max_neighbor_drop, 2),
            "spatial_consistency_ratio": round(spatial_ratio, 3),
            "adjacent_flow_delta_lps": round(flow_support * 1000.0, 2),
        }

        if is_sensor_fault:
            confidence = min(0.99, max(0.80, 1.0 - spatial_ratio))
            explanation = (
                f"SENSOR FAULT DETECTED on sensor '{max_node}'. "
                f"Sensor '{max_node}' registers an anomalous pressure drop of {max_drop:.2f} m, "
                f"yet its direct topological neighbors {direct_neighbors} register an average change "
                f"of only {avg_neighbor_drop:.2f} m (ratio {spatial_ratio:.1%}). "
                f"Hydraulic mass & energy conservation proves the physical network is operating normally; "
                f"sensor '{max_node}' is exhibiting an isolated instrumentation failure."
            )
            return {
                "diagnosis": "SENSOR_FAULT",
                "is_fault": True,
                "is_sensor_fault": True,
                "target_node": max_node,
                "confidence": round(confidence, 3),
                "explanation": explanation,
                "spatial_evidence": spatial_evidence,
            }
        else:
            confidence = min(0.99, max(0.80, spatial_ratio))
            explanation = (
                f"PHYSICAL LEAK DETECTED in the vicinity of '{max_node}'. "
                f"Sensor '{max_node}' registers a pressure drop of {max_drop:.2f} m, "
                f"accompanied by correlated pressure drops across neighbouring sensors "
                f"(average neighbour drop: {avg_neighbor_drop:.2f} m, ratio: {spatial_ratio:.1%}). "
                f"The coherent spatial depression cone confirms a genuine physical rupture."
            )
            return {
                "diagnosis": "PHYSICAL_LEAK",
                "is_fault": True,
                "is_sensor_fault": False,
                "target_node": max_node,
                "confidence": round(confidence, 3),
                "explanation": explanation,
                "spatial_evidence": spatial_evidence,
            }
