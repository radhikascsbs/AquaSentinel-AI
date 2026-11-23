"""
AquaSentinel AI - Hydraulic Engine
Executes hydraulic simulations using real WNTR / EPANET engines,
exports pressure, flow, and demand datasets to CSV, and performs
detailed scenario comparisons.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
import os
import pandas as pd
import numpy as np
import wntr


@dataclass
class SimulationResults:
    """Container for hydraulic simulation outputs."""
    pressure: pd.DataFrame
    flowrate: pd.DataFrame
    demand: pd.DataFrame
    head: pd.DataFrame
    velocity: pd.DataFrame
    leak_demand: pd.DataFrame
    engine_type: str
    network_model: wntr.network.WaterNetworkModel


class HydraulicEngine:
    """
    High-level controller for water distribution network simulation
    supporting both WNTRSimulator (Pressure-Dependent Demand / Leaks)
    and EpanetSimulator.
    """

    def __init__(self, wn: wntr.network.WaterNetworkModel):
        self.wn = wn

    def run(
        self,
        engine: str = "WNTR",
        mode: str = "PDD",
    ) -> SimulationResults:
        """
        Executes hydraulic simulation.

        Parameters
        ----------
        engine : str
            Simulation engine to use: 'WNTR' (supports PDD and leaks)
            or 'EPANET' (uses native EPANET DLL).
        mode : str
            Demand model for WNTRSimulator: 'PDD' (Pressure Dependent Demand)
            or 'DD' (Demand Driven).

        Returns
        -------
        SimulationResults
            Object containing time-series DataFrames for pressure, flow, demand, etc.
        """
        engine_upper = engine.upper()

        if engine_upper == "EPANET":
            sim = wntr.sim.EpanetSimulator(self.wn)
            raw_res = sim.run_sim()
            leak_df = pd.DataFrame(
                0.0,
                index=raw_res.node["pressure"].index,
                columns=raw_res.node["pressure"].columns,
            )
        elif engine_upper == "WNTR":
            if mode:
                self.wn.options.hydraulic.demand_model = mode
            sim = wntr.sim.WNTRSimulator(self.wn)
            raw_res = sim.run_sim()
            # Extract leak_demand if available from WNTRSimulator
            if hasattr(raw_res, "node") and "leak_demand" in raw_res.node:
                leak_df = raw_res.node["leak_demand"]
            else:
                leak_df = pd.DataFrame(
                    0.0,
                    index=raw_res.node["pressure"].index,
                    columns=raw_res.node["pressure"].columns,
                )
        else:
            raise ValueError(f"Unsupported simulation engine: {engine}. Use 'WNTR' or 'EPANET'.")

        return SimulationResults(
            pressure=raw_res.node["pressure"],
            flowrate=raw_res.link["flowrate"],
            demand=raw_res.node["demand"],
            head=raw_res.node["head"],
            velocity=raw_res.link.get("velocity", pd.DataFrame()),
            leak_demand=leak_df,
            engine_type=engine_upper,
            network_model=self.wn,
        )

    @staticmethod
    def save_to_csv(
        results: SimulationResults,
        output_dir: str = "data/simulation_results",
        prefix: str = "normal",
    ) -> Dict[str, str]:
        """
        Saves pressure, flow, demand, and leak demand DataFrames to CSV files.

        Parameters
        ----------
        results : SimulationResults
            Simulation results to export.
        output_dir : str
            Directory to save CSV files into.
        prefix : str
            Prefix for file names (e.g., 'normal', 'leak_zone_b').

        Returns
        -------
        Dict[str, str]
            Dictionary mapping result category to absolute file path.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # 1. Pressure (m)
        p_path = out_path / f"{prefix}_pressure.csv"
        results.pressure.to_csv(p_path, index_label="timestamp_seconds")
        saved_files["pressure"] = str(p_path.resolve())

        # 2. Flowrate (m^3/s)
        f_path = out_path / f"{prefix}_flowrate.csv"
        results.flowrate.to_csv(f_path, index_label="timestamp_seconds")
        saved_files["flowrate"] = str(f_path.resolve())

        # 3. Consumer Demand (m^3/s)
        d_path = out_path / f"{prefix}_demand.csv"
        results.demand.to_csv(d_path, index_label="timestamp_seconds")
        saved_files["demand"] = str(d_path.resolve())

        # 4. Leak Demand (m^3/s)
        ld_path = out_path / f"{prefix}_leak_demand.csv"
        results.leak_demand.to_csv(ld_path, index_label="timestamp_seconds")
        saved_files["leak_demand"] = str(ld_path.resolve())

        # 5. Combined summary CSV: Average and snapshot metrics across all nodes
        summary_rows = []
        wn = results.network_model
        for node_name in results.pressure.columns:
            node = wn.get_node(node_name)
            node_type = getattr(node, "node_type", "Junction")
            zone = getattr(node, "zone", "N/A")
            pressures = results.pressure[node_name]
            demands = results.demand[node_name]
            leaks = results.leak_demand[node_name]

            summary_rows.append({
                "node_name": node_name,
                "node_type": node_type,
                "zone": zone,
                "mean_pressure_m": pressures.mean(),
                "min_pressure_m": pressures.min(),
                "max_pressure_m": pressures.max(),
                "mean_demand_lps": demands.mean() * 1000.0,
                "total_demand_m3": demands.sum() * 3600.0 if len(demands) > 1 else demands.iloc[0],
                "mean_leak_lps": leaks.mean() * 1000.0,
                "total_leak_m3": leaks.sum() * 3600.0 if len(leaks) > 1 else leaks.iloc[0],
            })

        summary_df = pd.DataFrame(summary_rows)
        s_path = out_path / f"{prefix}_node_summary.csv"
        summary_df.to_csv(s_path, index=False)
        saved_files["summary"] = str(s_path.resolve())

        return saved_files

    @staticmethod
    def compare_scenarios(
        normal: SimulationResults,
        leak: SimulationResults,
        leak_node: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Performs rigorous mathematical comparison between baseline and leak simulations.
        """
        wn = normal.network_model

        # 1. Node-by-Node Pressure & Demand Delta
        # Match timesteps that are common to both simulations
        common_times = normal.pressure.index.intersection(leak.pressure.index)
        if len(common_times) == 0:
            # If timesteps differ, use final or first timestamp
            p_norm = normal.pressure.iloc[-1]
            p_leak = leak.pressure.iloc[-1]
            d_norm = normal.demand.iloc[-1]
            d_leak = leak.demand.iloc[-1]
            l_leak = leak.leak_demand.iloc[-1]
        else:
            p_norm = normal.pressure.loc[common_times].mean()
            p_leak = leak.pressure.loc[common_times].mean()
            d_norm = normal.demand.loc[common_times].mean()
            d_leak = leak.demand.loc[common_times].mean()
            l_leak = leak.leak_demand.loc[common_times].mean()

        node_diff_rows = []
        for n in normal.pressure.columns:
            node = wn.get_node(n)
            zone = getattr(node, "zone", "N/A")
            is_junction = n in wn.junction_name_list
            if not is_junction:
                continue

            pn = p_norm[n]
            pl = p_leak[n]
            p_drop = pn - pl
            p_drop_pct = (p_drop / pn * 100.0) if pn > 1e-4 else 0.0

            ql = l_leak[n] * 1000.0  # L/s
            dn = d_norm[n] * 1000.0  # L/s
            dl = d_leak[n] * 1000.0  # L/s

            node_diff_rows.append({
                "junction": n,
                "zone": zone,
                "normal_pressure_m": round(pn, 2),
                "leak_pressure_m": round(pl, 2),
                "pressure_drop_m": round(p_drop, 2),
                "pressure_drop_pct": round(p_drop_pct, 1),
                "normal_demand_lps": round(dn, 2),
                "leak_demand_lps": round(dl, 2),
                "leak_discharge_lps": round(ql, 2),
                "is_leak_origin": (n == leak_node),
            })

        node_df = pd.DataFrame(node_diff_rows).sort_values("pressure_drop_m", ascending=False)

        # 2. Zone-Level Pressure Drop Aggregation
        zone_summary = (
            node_df.groupby("zone")
            .agg(
                avg_normal_pressure_m=("normal_pressure_m", "mean"),
                avg_leak_pressure_m=("leak_pressure_m", "mean"),
                avg_pressure_drop_m=("pressure_drop_m", "mean"),
                max_pressure_drop_m=("pressure_drop_m", "max"),
                total_leak_discharge_lps=("leak_discharge_lps", "sum"),
            )
            .round(2)
            .reset_index()
        )

        # 3. System-Wide Aggregated Totals
        total_norm_demand_lps = sum(d_norm[j] * 1000.0 for j in wn.junction_name_list)
        total_leak_demand_lps = sum(d_leak[j] * 1000.0 for j in wn.junction_name_list)
        total_leak_loss_lps = sum(l_leak[j] * 1000.0 for j in wn.junction_name_list)

        system_summary = {
            "total_normal_demand_lps": round(total_norm_demand_lps, 2),
            "total_actual_consumer_demand_lps": round(total_leak_demand_lps, 2),
            "total_leak_water_loss_lps": round(total_leak_loss_lps, 2),
            "system_demand_increase_pct": round(
                ((total_leak_demand_lps + total_leak_loss_lps - total_norm_demand_lps) / total_norm_demand_lps * 100.0)
                if total_norm_demand_lps > 0 else 0.0,
                1,
            ),
            "max_system_pressure_drop_m": round(node_df["pressure_drop_m"].max(), 2),
            "worst_affected_node": node_df.iloc[0]["junction"],
        }

        # 4. Pipe Flow Comparison (Top 5 links by flow change)
        pipe_diff_rows = []
        f_norm = normal.flowrate.mean()
        f_leak = leak.flowrate.mean()
        for p in normal.flowrate.columns:
            fn = f_norm[p] * 1000.0  # L/s
            fl = f_leak[p] * 1000.0  # L/s
            diff = fl - fn
            pipe_diff_rows.append({
                "pipe": p,
                "normal_flow_lps": round(fn, 2),
                "leak_flow_lps": round(fl, 2),
                "flow_increase_lps": round(diff, 2),
                "flow_increase_pct": round((diff / abs(fn) * 100.0) if abs(fn) > 1e-4 else 0.0, 1),
            })
        pipe_df = pd.DataFrame(pipe_diff_rows).sort_values("flow_increase_lps", ascending=False)

        return {
            "node_comparison": node_df,
            "zone_summary": zone_summary,
            "system_summary": system_summary,
            "pipe_comparison": pipe_df,
        }
