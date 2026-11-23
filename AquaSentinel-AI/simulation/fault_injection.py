"""
AquaSentinel AI - Fault Injection Engine
Injects physical hydraulic anomalies (burst leaks, pipe ruptures, valve failures)
into the WNTR water network model using valid native WNTR leak formulations.
"""

from typing import Dict, Any, List, Optional
import math
import wntr


class FaultInjector:
    """
    Manages hydraulic anomaly and leak injections on junctions and pipes.
    """

    @staticmethod
    def inject_junction_leak(
        wn: wntr.network.WaterNetworkModel,
        junction_name: str,
        leak_area: float = 0.002,
        discharge_coeff: float = 0.75,
        start_time: Optional[int] = 0,
        end_time: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Adds a native pressure-dependent leak to a junction in the WNTR network.

        The leak discharge is modeled via the orifice equation:
            Q_leak = discharge_coeff * leak_area * sqrt(2 * g * h)

        Parameters
        ----------
        wn : wntr.network.WaterNetworkModel
            The target network model.
        junction_name : str
            Name of junction to leak (must exist in model).
        leak_area : float
            Leak orifice cross-sectional area in m^2 (e.g. 0.002 m^2 ~ 50mm diameter).
        discharge_coeff : float
            Orifice discharge coefficient (default 0.75).
        start_time : int, optional
            Time in seconds when leak begins (default 0).
        end_time : int, optional
            Time in seconds when leak is repaired (default None = continuous).

        Returns
        -------
        Dict[str, Any]
            Metadata record of the injected leak fault.
        """
        if junction_name not in wn.junction_name_list:
            raise ValueError(f"Junction '{junction_name}' does not exist in network model.")

        junction = wn.get_node(junction_name)
        zone = getattr(junction, "zone", "Unknown")

        # Native WNTR leak injection
        junction.add_leak(
            wn=wn,
            area=leak_area,
            discharge_coeff=discharge_coeff,
            start_time=start_time,
            end_time=end_time,
        )

        equivalent_diameter_mm = 2.0 * math.sqrt(leak_area / math.pi) * 1000.0

        metadata = {
            "fault_type": "junction_leak",
            "target": junction_name,
            "zone": zone,
            "leak_area_m2": leak_area,
            "equivalent_diameter_mm": round(equivalent_diameter_mm, 1),
            "discharge_coeff": discharge_coeff,
            "start_time_seconds": start_time,
            "end_time_seconds": end_time,
            "status": "ACTIVE",
        }
        return metadata

    @classmethod
    def inject_zone_b_leak(
        cls,
        wn: wntr.network.WaterNetworkModel,
        junction_name: str = "J_B2",
        leak_area: float = 0.002,
        discharge_coeff: float = 0.75,
        start_time: Optional[int] = 0,
        end_time: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Specifically targets and injects a leak in Zone B to satisfy user requirements.
        """
        node = wn.get_node(junction_name)
        zone = getattr(node, "zone", "")
        if "B" not in zone:
            raise ValueError(
                f"Selected junction '{junction_name}' is in zone '{zone}', not in Zone B!"
            )
        return cls.inject_junction_leak(
            wn=wn,
            junction_name=junction_name,
            leak_area=leak_area,
            discharge_coeff=discharge_coeff,
            start_time=start_time,
            end_time=end_time,
        )

    @staticmethod
    def inject_pipe_leak(
        wn: wntr.network.WaterNetworkModel,
        pipe_name: str,
        leak_area: float = 0.002,
        discharge_coeff: float = 0.75,
        start_time: Optional[int] = 0,
        end_time: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Injects a leak along a pipe by splitting the pipe into two segments
        with an intermediate leak node (valid WNTR pipe leak method).
        """
        if pipe_name not in wn.pipe_name_list:
            raise ValueError(f"Pipe '{pipe_name}' does not exist in network model.")

        pipe = wn.get_link(pipe_name)
        start_node = pipe.start_node_name
        end_node = pipe.end_node_name
        orig_length = pipe.length
        orig_diameter = pipe.diameter
        orig_roughness = pipe.roughness

        # Intermediate leak junction name
        leak_junc_name = f"{pipe_name}_leak_node"
        new_pipe_name = f"{pipe_name}_seg2"

        # Split pipe using WNTR morph utility in-place
        wntr.morph.split_pipe(
            wn=wn,
            pipe_name_to_split=pipe_name,
            new_pipe_name=new_pipe_name,
            new_junction_name=leak_junc_name,
            return_copy=False,
        )

        leak_node = wn.get_node(leak_junc_name)
        # Inherit zone from start node
        parent_zone = getattr(wn.get_node(start_node), "zone", "Zone_B")
        leak_node.zone = parent_zone
        leak_node.tag = parent_zone

        # Add native leak to the split node
        leak_node.add_leak(
            wn=wn,
            area=leak_area,
            discharge_coeff=discharge_coeff,
            start_time=start_time,
            end_time=end_time,
        )

        equivalent_diameter_mm = 2.0 * math.sqrt(leak_area / math.pi) * 1000.0

        return {
            "fault_type": "pipe_leak",
            "target_pipe": pipe_name,
            "created_leak_node": leak_junc_name,
            "zone": parent_zone,
            "leak_area_m2": leak_area,
            "equivalent_diameter_mm": round(equivalent_diameter_mm, 1),
            "discharge_coeff": discharge_coeff,
            "start_time_seconds": start_time,
            "end_time_seconds": end_time,
            "status": "ACTIVE",
        }

    @staticmethod
    def set_isolation_valve(
        wn: wntr.network.WaterNetworkModel,
        valve_name: str,
        open_valve: bool = False,
    ) -> Dict[str, Any]:
        """
        Controls the status of an isolation valve (OPEN or CLOSED).
        """
        if valve_name not in wn.valve_name_list:
            raise ValueError(f"Valve '{valve_name}' does not exist in network model.")

        valve = wn.get_link(valve_name)
        new_status = wntr.network.LinkStatus.Open if open_valve else wntr.network.LinkStatus.Closed
        valve.initial_status = new_status
        return {
            "action": "valve_control",
            "valve": valve_name,
            "status": "OPEN" if open_valve else "CLOSED",
        }

    @staticmethod
    def get_active_leaks(wn: wntr.network.WaterNetworkModel) -> List[Dict[str, Any]]:
        """
        Returns list of all nodes currently configured with leaks.
        """
        leaks = []
        for j_name in wn.junction_name_list:
            node = wn.get_node(j_name)
            if getattr(node, "_leak", False):
                leaks.append({
                    "junction": j_name,
                    "zone": getattr(node, "zone", "Unknown"),
                    "area_m2": getattr(node, "_leak_area", 0.0),
                    "discharge_coeff": getattr(node, "_leak_discharge_coeff", 0.75),
                })
        return leaks
