"""
AquaSentinel AI - Decision Engine: Action Optimizer
Simulates and evaluates candidate valve closures on network copies, optimizes the
tradeoff between leak mitigation and customer service continuity, and executes
automated or gated simulated valve actions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import copy
import numpy as np
import pandas as pd
import wntr

from simulation.hydraulic_engine import HydraulicEngine


@dataclass
class CandidateActionEvaluation:
    """Evaluation metrics for a candidate valve action tested on a network copy."""
    action_id: str
    action_name: str
    target_links: List[str]
    remaining_leak_lps: float
    delivered_demand_lps: float
    unmet_demand_lps: float
    customer_service_pct: float
    pressure_violations_count: int  # Junctions with P < 15.0 m
    min_system_pressure_m: float
    mean_system_pressure_m: float
    zones_affected_count: int
    affected_zones: List[str]
    optimization_penalty: float
    is_best_action: bool = False


@dataclass
class OptimizationResult:
    """Complete output of the Action Optimizer."""
    decision: str                      # 'AUTO_ACT' or 'NO_ACT_MANUAL_INSPECTION'
    auto_action_approved: bool
    policy_reason: str
    target_suspect_pipe: str
    ai_confidence: float
    confidence_threshold: float
    sensor_trustworthy: bool
    evaluated_candidates: List[CandidateActionEvaluation]
    selected_action: Optional[CandidateActionEvaluation] = None
    applied_simulation_model: Optional[wntr.network.WaterNetworkModel] = None
    action_log: Optional[Dict[str, Any]] = None


class ActionOptimizer:
    """
    Evaluates candidate valve and segment isolations on network copies, selects
    the optimal mitigation plan, and applies simulated valve actions when AI confidence
    and sensor integrity safety gates are satisfied.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.90,
        pressure_service_standard_m: float = 15.0,
    ):
        self.confidence_threshold = confidence_threshold
        self.pressure_service_standard_m = pressure_service_standard_m

    def _generate_candidate_actions(
        self,
        wn: wntr.network.WaterNetworkModel,
        suspect_pipe: str,
    ) -> List[Dict[str, Any]]:
        """
        Formulates candidate valve and pipe isolation strategies based on
        the network topology surrounding the suspect pipe.
        """
        candidates = []

        # Candidate 0: Baseline - Do Nothing (keep all current valves as-is)
        candidates.append({
            "id": "ACTION_DO_NOTHING",
            "name": "No Action (Keep Current Valves)",
            "target_links": [],
        })

        # Candidate 1: Direct Segment Isolation of Suspect Pipe
        # In modern DMA distribution networks, individual pipes possess boundary gate valves.
        pipe_targets = [suspect_pipe]
        # If the pipe was split (e.g. morph split), include the twin segment
        seg2_name = f"{suspect_pipe}_seg2"
        if seg2_name in wn.link_name_list:
            pipe_targets.append(seg2_name)

        candidates.append({
            "id": "ACTION_ISOLATE_SUSPECT_PIPE",
            "name": f"Isolate Pipe Segment '{suspect_pipe}'",
            "target_links": pipe_targets,
        })

        # Candidate 2: Major District Feeder Valve Closures
        if "VALVE_ISO_B" in wn.link_name_list:
            candidates.append({
                "id": "ACTION_CLOSE_VALVE_ISO_B",
                "name": "Close Feeder Valve 'VALVE_ISO_B' (Zone B Inflow)",
                "target_links": ["VALVE_ISO_B"],
            })

        if "VALVE_ISO_C" in wn.link_name_list:
            candidates.append({
                "id": "ACTION_CLOSE_VALVE_ISO_C",
                "name": "Close Transfer Valve 'VALVE_ISO_C' (Zone B-C Inflow)",
                "target_links": ["VALVE_ISO_C"],
            })

        if "VALVE_ISO_A" in wn.link_name_list:
            candidates.append({
                "id": "ACTION_CLOSE_VALVE_ISO_A",
                "name": "Close Feeder Valve 'VALVE_ISO_A' (Zone A Core)",
                "target_links": ["VALVE_ISO_A"],
            })

        # Candidate 3: Boundary Cross-Links / Adjacent Ring Closures
        if suspect_pipe in wn.pipe_name_list:
            pipe = wn.get_link(suspect_pipe)
            u = pipe.start_node_name
            v = pipe.end_node_name
            # Identify other links incident to u or v
            adjacent_links = []
            for l_name, link in wn.links():
                if l_name != suspect_pipe and (link.start_node_name in (u, v) or link.end_node_name in (u, v)):
                    if link.link_type == "Pipe":
                        adjacent_links.append(l_name)

            if adjacent_links:
                first_adj = adjacent_links[0]
                candidates.append({
                    "id": f"ACTION_ISOLATE_ADJACENT_{first_adj}",
                    "name": f"Isolate Adjacent Pipe '{first_adj}'",
                    "target_links": [first_adj],
                })

        return candidates

    def _evaluate_candidate(
        self,
        base_wn: wntr.network.WaterNetworkModel,
        candidate_def: Dict[str, Any],
        required_demands: Dict[str, float],
    ) -> CandidateActionEvaluation:
        """
        Runs hydraulic simulation on a pristine COPY of the network with the candidate
        closure applied, and extracts performance indicators.
        """
        # Step 1: Work on a strict isolated COPY of the network model
        wn_copy = copy.deepcopy(base_wn)

        # Apply candidate closures on the copy
        for link_name in candidate_def["target_links"]:
            if link_name in wn_copy.link_name_list:
                link = wn_copy.get_link(link_name)
                link.initial_status = "Closed"

        # Step 2: Re-run the real WNTR simulation
        engine = HydraulicEngine(wn_copy)
        results = engine.run(engine="WNTR", mode="PDD")

        # Step 3: Measure performance indicators
        # A. Remaining water loss
        total_leak_m3s = sum(
            float(results.leak_demand[j].iloc[0])
            for j in wn_copy.junction_name_list
            if j in results.leak_demand.columns
        )
        total_leak_lps = total_leak_m3s * 1000.0

        # B. Delivered consumer demand on customer junctions (exclude reservoirs/tanks)
        delivered_demands = {
            j: float(results.demand[j].iloc[0]) * 1000.0
            for j in wn_copy.junction_name_list
        }
        total_delivered_lps = sum(max(0.0, d) for d in delivered_demands.values())
        total_required_lps = sum(required_demands.values())

        # C. Unmet demand
        unmet_per_junction = {
            j: max(0.0, required_demands.get(j, 0.0) - delivered_demands.get(j, 0.0))
            for j in wn_copy.junction_name_list
        }
        total_unmet_lps = sum(unmet_per_junction.values())
        customer_service_pct = (
            (total_delivered_lps / total_required_lps * 100.0)
            if total_required_lps > 1e-4 else 100.0
        )

        # D. Pressure problems
        junction_pressures = {
            j: float(results.pressure[j].iloc[0])
            for j in wn_copy.junction_name_list
        }
        pressure_violations = sum(
            1 for p in junction_pressures.values()
            if p < self.pressure_service_standard_m
        )
        min_p = min(junction_pressures.values()) if junction_pressures else 0.0
        mean_p = float(np.mean(list(junction_pressures.values()))) if junction_pressures else 0.0

        # E. Zones affected
        affected_zones_set = set()
        for j, p in junction_pressures.items():
            node = wn_copy.get_node(j)
            z = getattr(node, "zone", "Unassigned")
            if p < self.pressure_service_standard_m or unmet_per_junction.get(j, 0.0) > 0.5:
                affected_zones_set.add(z)

        # F. Multi-Objective Cost Function:
        # Heavily penalizes remaining water loss (leak) while rewarding customer service
        # Penalty = (10.0 * Leak_L/s) + (5.0 * Unmet_L/s) + (15.0 * Violations) + (25.0 * Zones_Affected)
        optimization_penalty = (
            (10.0 * total_leak_lps)
            + (5.0 * total_unmet_lps)
            + (15.0 * float(pressure_violations))
            + (25.0 * float(len(affected_zones_set)))
        )

        return CandidateActionEvaluation(
            action_id=candidate_def["id"],
            action_name=candidate_def["name"],
            target_links=candidate_def["target_links"],
            remaining_leak_lps=round(total_leak_lps, 2),
            delivered_demand_lps=round(total_delivered_lps, 2),
            unmet_demand_lps=round(total_unmet_lps, 2),
            customer_service_pct=round(customer_service_pct, 1),
            pressure_violations_count=pressure_violations,
            min_system_pressure_m=round(min_p, 2),
            mean_system_pressure_m=round(mean_p, 2),
            zones_affected_count=len(affected_zones_set),
            affected_zones=sorted(list(affected_zones_set)),
            optimization_penalty=round(optimization_penalty, 2),
        )

    def optimize_action(
        self,
        wn: wntr.network.WaterNetworkModel,
        top_suspect_pipe: str,
        ai_confidence: float,
        sensor_trustworthy: bool,
    ) -> OptimizationResult:
        """
        Executes multi-objective action optimization and applies the safety gate.

        Parameters
        ----------
        wn : wntr.network.WaterNetworkModel
            Water network model currently under fault/leak condition.
        top_suspect_pipe : str
            Suspect pipe identified by fault localisation.
        ai_confidence : float
            Confidence score from fault classifier (0.0 to 1.0).
        sensor_trustworthy : bool
            True if sensor diagnostics confirm physical leak (no sensor fault).

        Returns
        -------
        OptimizationResult
            Optimization evaluation, decision path, and applied simulated action.
        """
        # Baseline customer demand requirement under normal operating conditions
        # (accounts for diurnal pattern multipliers at the active simulation timestep)
        wn_baseline = copy.deepcopy(wn)
        for j_name in wn_baseline.junction_name_list:
            node = wn_baseline.get_node(j_name)
            if hasattr(node, '_leak'):
                node._leak = False
        res_baseline = HydraulicEngine(wn_baseline).run()
        required_demands = {
            j: max(0.0, float(res_baseline.demand[j].iloc[0]) * 1000.0)
            for j in wn_baseline.junction_name_list
        }

        # Step 1: Generate candidate valve closures
        candidate_defs = self._generate_candidate_actions(wn, top_suspect_pipe)

        # Step 2 & 3: Try each candidate closure on a COPY of the network & measure metrics
        evaluations: List[CandidateActionEvaluation] = []
        for c_def in candidate_defs:
            eval_result = self._evaluate_candidate(wn, c_def, required_demands)
            evaluations.append(eval_result)

        # Step 4: Pick the best option: stop the leak while keeping customer service
        # Best option has lowest optimization penalty
        best_eval = min(evaluations, key=lambda x: x.optimization_penalty)
        best_eval.is_best_action = True

        # Safety Gate & Automatic Valve Response Check:
        # Only act automatically when:
        # 1. AI confidence is above threshold (e.g. 90%)
        # 2. Sensor-fault check confirms sensor readings are trustworthy
        is_confidence_sufficient = (ai_confidence >= self.confidence_threshold)
        is_safe_to_auto_act = is_confidence_sufficient and sensor_trustworthy

        if not is_confidence_sufficient:
            decision = "NO_ACT_MANUAL_INSPECTION"
            reason = (
                f"AI classification confidence ({ai_confidence * 100.0:.1f}%) is below the automated "
                f"action threshold ({self.confidence_threshold * 100.0:.1f}%). "
                f"Automated valve intervention inhibited to prevent unintended supply interruption. "
                f"RECOMMENDATION: Dispatch field crew for manual acoustic & visual pipe inspection."
            )
            return OptimizationResult(
                decision=decision,
                auto_action_approved=False,
                policy_reason=reason,
                target_suspect_pipe=top_suspect_pipe,
                ai_confidence=ai_confidence,
                confidence_threshold=self.confidence_threshold,
                sensor_trustworthy=sensor_trustworthy,
                evaluated_candidates=evaluations,
                selected_action=best_eval,
                applied_simulation_model=None,
                action_log=None,
            )

        if not sensor_trustworthy:
            decision = "NO_ACT_MANUAL_INSPECTION"
            reason = (
                f"Sensor integrity diagnostics detected instrumentation failure. "
                f"Telemetry is NOT trustworthy. Automated closure strictly inhibited. "
                f"RECOMMENDATION: Dispatch instrumentation technician to recalibrate / replace faulty sensor."
            )
            return OptimizationResult(
                decision=decision,
                auto_action_approved=False,
                policy_reason=reason,
                target_suspect_pipe=top_suspect_pipe,
                ai_confidence=ai_confidence,
                confidence_threshold=self.confidence_threshold,
                sensor_trustworthy=sensor_trustworthy,
                evaluated_candidates=evaluations,
                selected_action=best_eval,
                applied_simulation_model=None,
                action_log=None,
            )

        # Step 5: High Confidence + Trustworthy Sensors -> AUTO-ACT
        # Apply the best option to the active simulation network model
        decision = "AUTO_ACT"
        reason = (
            f"AI confidence ({ai_confidence * 100.0:.1f}%) meets/exceeds safety threshold "
            f"({self.confidence_threshold * 100.0:.1f}%) and hydraulic sensors are validated trustworthy. "
            f"AUTOMATED SIMULATED VALVE RESPONSE EXECUTED."
        )

        applied_wn = copy.deepcopy(wn)
        for link_name in best_eval.target_links:
            if link_name in applied_wn.link_name_list:
                link = applied_wn.get_link(link_name)
                link.initial_status = "Closed"

        action_log = {
            "event_type": "SIMULATED_VALVE_ACTION",
            "timestamp": datetime.now().isoformat(),
            "target_suspect_pipe": top_suspect_pipe,
            "selected_action": best_eval.action_name,
            "isolated_links": best_eval.target_links,
            "ai_confidence_pct": round(ai_confidence * 100.0, 1),
            "expected_water_loss_reduction_pct": round(
                (1.0 - (best_eval.remaining_leak_lps / evaluations[0].remaining_leak_lps)) * 100.0
                if evaluations[0].remaining_leak_lps > 0 else 0.0,
                1,
            ),
            "customer_service_retained_pct": best_eval.customer_service_pct,
            "system_status": "ONLINE_ACTIVE",
        }

        return OptimizationResult(
            decision=decision,
            auto_action_approved=True,
            policy_reason=reason,
            target_suspect_pipe=top_suspect_pipe,
            ai_confidence=ai_confidence,
            confidence_threshold=self.confidence_threshold,
            sensor_trustworthy=sensor_trustworthy,
            evaluated_candidates=evaluations,
            selected_action=best_eval,
            applied_simulation_model=applied_wn,
            action_log=action_log,
        )
