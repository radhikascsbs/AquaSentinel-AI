"""
AquaSentinel AI - Decision Engine: Recovery Verification
Simulates post-mitigation hydraulics, verifies physical recovery, compares
before vs after metrics (water loss, pressure, unmet demand), and evaluates
whether any hydraulic anomalies remain in the distribution network.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import copy
import numpy as np
import pandas as pd
import wntr

from simulation.hydraulic_engine import HydraulicEngine


@dataclass
class RecoveryReport:
    """Comprehensive before-vs-after verification report."""
    status: str                         # 'RECOVERY_VERIFIED_SUCCESSFUL', 'PARTIAL_RECOVERY', 'UNSUCCESSFUL'
    target_pipe: str
    action_description: str
    water_loss_before_lps: float
    water_loss_after_lps: float
    water_loss_reduction_pct: float
    delivered_demand_before_lps: float
    delivered_demand_after_lps: float
    unmet_demand_before_lps: float
    unmet_demand_after_lps: float
    customer_service_pct_before: float
    customer_service_pct_after: float
    mean_pressure_before_m: float
    mean_pressure_after_m: float
    min_pressure_before_m: float
    min_pressure_after_m: float
    pressure_violations_before: int
    pressure_violations_after: int
    zone_status_table: pd.DataFrame
    remaining_abnormalities: List[str]
    is_network_operational: bool
    action_log: Optional[Dict[str, Any]] = None


class RecoveryVerifier:
    """
    Simulates and validates post-mitigation network state against baseline fault state.
    """

    def __init__(self, pressure_service_standard_m: float = 15.0):
        self.pressure_service_standard_m = pressure_service_standard_m

    def verify_recovery(
        self,
        pre_wn: wntr.network.WaterNetworkModel,
        post_wn: wntr.network.WaterNetworkModel,
        target_pipe: str,
        action_log: Optional[Dict[str, Any]] = None,
    ) -> RecoveryReport:
        """
        Executes simulations on pre-mitigation and post-mitigation models,
        measuring recovery metrics and detecting residual abnormalities.
        """
        # 1. Run real simulations on both states
        pre_engine = HydraulicEngine(pre_wn)
        pre_results = pre_engine.run(engine="WNTR", mode="PDD")

        post_engine = HydraulicEngine(post_wn)
        post_results = post_engine.run(engine="WNTR", mode="PDD")

        # 2. Water Loss Metrics
        pre_leak_lps = sum(
            float(pre_results.leak_demand[j].iloc[0]) * 1000.0
            for j in pre_wn.junction_name_list
            if j in pre_results.leak_demand.columns
        )
        post_leak_lps = sum(
            float(post_results.leak_demand[j].iloc[0]) * 1000.0
            for j in post_wn.junction_name_list
            if j in post_results.leak_demand.columns
        )

        leak_reduction_pct = (
            ((pre_leak_lps - post_leak_lps) / pre_leak_lps * 100.0)
            if pre_leak_lps > 1e-4 else 100.0
        )

        # 3. Demand & Service Metrics
        wn_baseline = copy.deepcopy(pre_wn)
        for j_name in wn_baseline.junction_name_list:
            node = wn_baseline.get_node(j_name)
            if hasattr(node, '_leak'):
                node._leak = False
        res_baseline = HydraulicEngine(wn_baseline).run()
        required_demands = {
            j: max(0.0, float(res_baseline.demand[j].iloc[0]) * 1000.0)
            for j in wn_baseline.junction_name_list
        }
        total_req_lps = sum(required_demands.values())

        pre_delivered_lps = sum(
            max(0.0, float(pre_results.demand[j].iloc[0]) * 1000.0)
            for j in pre_wn.junction_name_list
        )
        post_delivered_lps = sum(
            max(0.0, float(post_results.demand[j].iloc[0]) * 1000.0)
            for j in post_wn.junction_name_list
        )

        pre_unmet_lps = max(0.0, total_req_lps - pre_delivered_lps)
        post_unmet_lps = max(0.0, total_req_lps - post_delivered_lps)

        pre_svc_pct = (pre_delivered_lps / total_req_lps * 100.0) if total_req_lps > 0 else 100.0
        post_svc_pct = (post_delivered_lps / total_req_lps * 100.0) if total_req_lps > 0 else 100.0

        # 4. Pressure Metrics
        pre_pressures = {j: float(pre_results.pressure[j].iloc[0]) for j in pre_wn.junction_name_list}
        post_pressures = {j: float(post_results.pressure[j].iloc[0]) for j in post_wn.junction_name_list}

        mean_p_pre = float(np.mean(list(pre_pressures.values())))
        mean_p_post = float(np.mean(list(post_pressures.values())))
        min_p_pre = min(pre_pressures.values())
        min_p_post = min(post_pressures.values())

        viol_pre = sum(1 for p in pre_pressures.values() if p < self.pressure_service_standard_m)
        viol_post = sum(1 for p in post_pressures.values() if p < self.pressure_service_standard_m)

        # 5. Zone Service Breakdown
        zones = ["Zone_A", "Zone_B", "Zone_C"]
        zone_records = []
        for z in zones:
            z_nodes = [j for j in pre_wn.junction_name_list if getattr(pre_wn.get_node(j), "zone", "") == z]
            if not z_nodes:
                continue

            z_req = sum(required_demands.get(j, 0.0) for j in z_nodes)
            z_deliv_pre = sum(max(0.0, float(pre_results.demand[j].iloc[0]) * 1000.0) for j in z_nodes)
            z_deliv_post = sum(max(0.0, float(post_results.demand[j].iloc[0]) * 1000.0) for j in z_nodes)
            z_p_pre = float(np.mean([pre_pressures[j] for j in z_nodes]))
            z_p_post = float(np.mean([post_pressures[j] for j in z_nodes]))

            z_svc_pct = (z_deliv_post / z_req * 100.0) if z_req > 0 else 100.0
            status_desc = "HEALTHY (Full Service)" if z_svc_pct >= 95.0 and z_p_post >= self.pressure_service_standard_m else "RESTRICTED (Service Reduced)"

            zone_records.append({
                "zone": z,
                "customer_demand_req_lps": round(z_req, 2),
                "delivered_pre_lps": round(z_deliv_pre, 2),
                "delivered_post_lps": round(z_deliv_post, 2),
                "avg_pressure_pre_m": round(z_p_pre, 2),
                "avg_pressure_post_m": round(z_p_post, 2),
                "service_continuity_pct": round(z_svc_pct, 1),
                "operational_status": status_desc,
            })

        zone_df = pd.DataFrame(zone_records)

        # 6. Residual Abnormality Inspection
        abnormalities = []
        # Check closed links
        closed_links = [
            l_name for l_name, l in post_wn.links()
            if getattr(l, "status", None) == wntr.network.LinkStatus.Closed or getattr(l, "initial_status", None) == wntr.network.LinkStatus.Closed
        ]
        if closed_links:
            abnormalities.append(
                f"SIMULATED ISOLATION: Link(s) {closed_links} actively closed to contain leak."
            )

        # Check if any junction has low or zero pressure
        low_p_junctions = [
            j for j, p in post_pressures.items()
            if p < self.pressure_service_standard_m
        ]
        if low_p_junctions:
            abnormalities.append(
                f"LOCALIZED PRESSURE ATTENUATION: Junction(s) {low_p_junctions} register pressure below "
                f"nominal standard ({self.pressure_service_standard_m}m) due to segment isolation."
            )
        else:
            abnormalities.append(
                "NOMINAL PRESSURE VERIFIED: All active junctions meet or exceed service pressure standard."
            )

        if post_leak_lps > 0.5:
            abnormalities.append(
                f"RESIDUAL LEAKAGE DETECTED: {post_leak_lps:.2f} L/s remains unisolated."
            )
        else:
            abnormalities.append(
                "COMPLETE LEAK CONTAINMENT: Leak discharge successfully eliminated to 0.00 L/s."
            )

        # Network operational status
        is_operational = (post_svc_pct >= 75.0) and (post_delivered_lps > 0.0)

        # Overall Status
        if leak_reduction_pct >= 85.0 and is_operational:
            status = "RECOVERY_VERIFIED_SUCCESSFUL"
        elif leak_reduction_pct >= 50.0:
            status = "PARTIAL_RECOVERY"
        else:
            status = "UNSUCCESSFUL"

        action_desc = action_log.get("selected_action", "Custom Isolation Action") if action_log else "Simulated Valve Mitigation"

        return RecoveryReport(
            status=status,
            target_pipe=target_pipe,
            action_description=action_desc,
            water_loss_before_lps=round(pre_leak_lps, 2),
            water_loss_after_lps=round(post_leak_lps, 2),
            water_loss_reduction_pct=round(leak_reduction_pct, 1),
            delivered_demand_before_lps=round(pre_delivered_lps, 2),
            delivered_demand_after_lps=round(post_delivered_lps, 2),
            unmet_demand_before_lps=round(pre_unmet_lps, 2),
            unmet_demand_after_lps=round(post_unmet_lps, 2),
            customer_service_pct_before=round(pre_svc_pct, 1),
            customer_service_pct_after=round(post_svc_pct, 1),
            mean_pressure_before_m=round(mean_p_pre, 2),
            mean_pressure_after_m=round(mean_p_post, 2),
            min_pressure_before_m=round(min_p_pre, 2),
            min_pressure_after_m=round(min_p_post, 2),
            pressure_violations_before=viol_pre,
            pressure_violations_after=viol_post,
            zone_status_table=zone_df,
            remaining_abnormalities=abnormalities,
            is_network_operational=is_operational,
            action_log=action_log,
        )
