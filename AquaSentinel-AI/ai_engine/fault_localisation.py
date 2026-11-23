"""
AquaSentinel AI - Topological Fault Localisation Engine
Uses NetworkX graph topology combined with nodal pressure drops and link flow deviations
to pinpoint and rank the top suspect pipes where a leak is occurring.
"""

from typing import Dict, List, Any, Optional
import networkx as nx
import numpy as np
import wntr


class FaultLocaliser:
    """
    Localises hydraulic leaks to specific pipes using NetworkX graph theory
    and differential pressure-flow analysis.
    """

    def __init__(self, wn: wntr.network.WaterNetworkModel):
        self.wn = wn
        self.graph = self._build_networkx_graph(wn)

    @staticmethod
    def _build_networkx_graph(wn: wntr.network.WaterNetworkModel) -> nx.Graph:
        """
        Extracts network topology into an undirected NetworkX graph with link attributes.
        """
        G = nx.Graph()

        # Add all nodes
        for name, node in wn.nodes():
            zone = getattr(node, "zone", "N/A")
            elev = getattr(node, "elevation", 0.0)
            coords = getattr(node, "coordinates", (0.0, 0.0))
            G.add_node(name, zone=zone, elevation=elev, pos=coords)

        # Add pipes as edges
        for p_name, pipe in wn.pipes():
            G.add_edge(
                pipe.start_node_name,
                pipe.end_node_name,
                name=p_name,
                link_type="pipe",
                length=pipe.length,
                diameter=pipe.diameter,
                weight=pipe.length,
            )

        # Add valves as edges
        for v_name, valve in wn.valves():
            G.add_edge(
                valve.start_node_name,
                valve.end_node_name,
                name=v_name,
                link_type="valve",
                length=10.0,
                diameter=valve.diameter,
                weight=10.0,
            )

        # Add pumps as edges
        for pump_name, pump in wn.pumps():
            G.add_edge(
                pump.start_node_name,
                pump.end_node_name,
                name=pump_name,
                link_type="pump",
                length=10.0,
                diameter=0.30,
                weight=10.0,
            )

        return G

    def rank_suspect_pipes(
        self,
        measured_pressures: Dict[str, float],
        normal_pressures: Dict[str, float],
        measured_flows: Optional[Dict[str, float]] = None,
        normal_flows: Optional[Dict[str, float]] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Ranks suspect pipes using differential pressure drops, flow deviations,
        and NetworkX topological graph distance.

        Parameters
        ----------
        measured_pressures : Dict[str, float]
            Observed pressures at each junction (m).
        normal_pressures : Dict[str, float]
            Baseline normal pressures at each junction (m).
        measured_flows : Dict[str, float], optional
            Observed flow rates on each link (L/s or m^3/s).
        normal_flows : Dict[str, float], optional
            Baseline normal flow rates on each link.
        top_k : int
            Number of top suspect pipes to return (default 3).

        Returns
        -------
        List[Dict[str, Any]]
            Ranked list of suspect pipes with likelihood score and hydraulic metrics.
        """
        # 1. Compute nodal pressure drops (positive = pressure dropped below normal)
        p_drops: Dict[str, float] = {}
        for j_name in self.wn.junction_name_list:
            p_meas = measured_pressures.get(j_name, normal_pressures.get(j_name, 0.0))
            if j_name in normal_pressures:
                p_norm = normal_pressures[j_name]
            else:
                # If this junction was created dynamically (e.g. split pipe leak node),
                # estimate normal baseline pressure from connected graph neighbors
                nbrs = [n for n in self.graph.neighbors(j_name) if n in normal_pressures]
                if nbrs:
                    p_norm = float(np.mean([normal_pressures[n] for n in nbrs]))
                else:
                    p_norm = p_meas
            p_drops[j_name] = max(0.0, float(p_norm - p_meas))

        if not p_drops or max(p_drops.values()) <= 1e-4:
            return []

        # Find the pressure drop epicenter (junction with highest pressure drop)
        epicenter_node = max(p_drops.items(), key=lambda x: x[1])[0]

        # 2. Compute shortest path graph distances from all nodes to epicenter
        try:
            shortest_hops = nx.single_source_shortest_path_length(self.graph, epicenter_node)
        except Exception:
            shortest_hops = {n: 1 for n in self.graph.nodes()}

        # 3. Compute link flow residuals (magnitude of flow change)
        q_deltas: Dict[str, float] = {}
        if measured_flows is not None and normal_flows is not None:
            for l_name in self.wn.link_name_list:
                qm = measured_flows.get(l_name, normal_flows.get(l_name, 0.0))
                qn = normal_flows.get(l_name, qm)
                q_deltas[l_name] = abs(float(qm - qn))

        max_q_delta = max(q_deltas.values()) if q_deltas and max(q_deltas.values()) > 1e-6 else 1.0
        max_p_drop = max(p_drops.values()) if max(p_drops.values()) > 1e-6 else 1.0

        # 4. Score each pipe in the network
        pipe_scores = []
        for p_name, pipe in self.wn.pipes():
            u = pipe.start_node_name
            v = pipe.end_node_name

            dp_u = p_drops.get(u, 0.0)
            dp_v = p_drops.get(v, 0.0)
            avg_endpoint_drop = (dp_u + dp_v) / 2.0
            max_endpoint_drop = max(dp_u, dp_v)

            # Topological proximity to the epicenter node
            hop_u = shortest_hops.get(u, 10)
            hop_v = shortest_hops.get(v, 10)
            min_hops = min(hop_u, hop_v)
            proximity_score = 1.0 / (1.0 + min_hops)

            # Flow change score
            dq = q_deltas.get(p_name, 0.0)
            flow_score = dq / max_q_delta

            # Endpoint pressure score
            pressure_score = (0.7 * (avg_endpoint_drop / max_p_drop)) + (0.3 * (max_endpoint_drop / max_p_drop))

            # Composite weighted suspect score:
            # 55% Pressure drop + 25% Flow disturbance + 20% Graph proximity
            if q_deltas:
                raw_score = (0.55 * pressure_score) + (0.25 * flow_score) + (0.20 * proximity_score)
            else:
                raw_score = (0.75 * pressure_score) + (0.25 * proximity_score)

            # Zone extraction
            zone_u = getattr(self.wn.get_node(u), "zone", "")
            zone_v = getattr(self.wn.get_node(v), "zone", "")
            pipe_zone = zone_u if zone_u == zone_v else f"{zone_u}/{zone_v}"

            pipe_scores.append({
                "pipe_name": p_name,
                "start_node": u,
                "end_node": v,
                "zone": pipe_zone,
                "raw_score": raw_score,
                "avg_pressure_drop_m": round(avg_endpoint_drop, 2),
                "max_endpoint_drop_m": round(max_endpoint_drop, 2),
                "flow_change": round(dq, 3),
                "hops_to_epicenter": min_hops,
            })

        # Consolidate split pipe segments (e.g. P_B1 and P_B1_seg2) back to parent pipe
        consolidated: Dict[str, Dict[str, Any]] = {}
        for item in pipe_scores:
            base_name = item["pipe_name"].split("_seg")[0]
            if base_name not in consolidated:
                item_copy = dict(item)
                item_copy["pipe_name"] = base_name
                consolidated[base_name] = item_copy
            else:
                existing = consolidated[base_name]
                if item["raw_score"] > existing["raw_score"]:
                    existing["raw_score"] = item["raw_score"]
                existing["avg_pressure_drop_m"] = max(existing["avg_pressure_drop_m"], item["avg_pressure_drop_m"])
                existing["max_endpoint_drop_m"] = max(existing["max_endpoint_drop_m"], item["max_endpoint_drop_m"])
                existing["flow_change"] = max(existing["flow_change"], item["flow_change"])
                existing["hops_to_epicenter"] = min(existing["hops_to_epicenter"], item["hops_to_epicenter"])
                if "_leak_node" in existing["start_node"]:
                    existing["start_node"] = item["start_node"] if "_leak_node" not in item["start_node"] else item["end_node"]
                if "_leak_node" in existing["end_node"]:
                    existing["end_node"] = item["end_node"] if "_leak_node" not in item["end_node"] else item["start_node"]

        pipe_scores = list(consolidated.values())

        # 5. Normalize scores to [0.0, 100.0]
        max_raw = max(x["raw_score"] for x in pipe_scores) if pipe_scores else 1.0
        for item in pipe_scores:
            item["suspect_score"] = round((item["raw_score"] / max_raw) * 100.0, 1)

        # Sort descending by suspect score
        pipe_scores.sort(key=lambda x: x["suspect_score"], reverse=True)

        # Assign ranks and select top_k
        top_suspects = []
        for rank, p in enumerate(pipe_scores[:top_k], start=1):
            p["rank"] = rank
            del p["raw_score"]
            top_suspects.append(p)

        return top_suspects
