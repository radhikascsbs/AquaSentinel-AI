"""
AquaSentinel AI - Water Network Builder
Builds a synthetic modular municipal water distribution network using WNTR.
"""

from typing import Dict, List, Any
import wntr


def build_network(
    duration_hours: int = 24,
    hydraulic_timestep_hours: int = 1,
    report_timestep_hours: int = 1,
) -> wntr.network.WaterNetworkModel:
    """
    Constructs a 3-zone water distribution network model with:
    - 1 reservoir (source)
    - 15 junctions distributed across 3 zones (A, B, C)
    - 17 transmission and distribution pipes
    - 2 pumps (primary lift pump and booster pump)
    - 3 isolation valves (TCV / control valves)
    - Junction demands with 24-hour diurnal patterns

    Parameters
    ----------
    duration_hours : int
        Total simulation duration in hours (default 24).
    hydraulic_timestep_hours : int
        Hydraulic calculation timestep in hours (default 1).
    report_timestep_hours : int
        Reporting timestep in hours (default 1).

    Returns
    -------
    wntr.network.WaterNetworkModel
        Configured water network model ready for simulation.
    """
    wn = wntr.network.WaterNetworkModel()

    # Configure simulation timing
    wn.options.time.duration = duration_hours * 3600
    wn.options.time.hydraulic_timestep = hydraulic_timestep_hours * 3600
    wn.options.time.report_timestep = report_timestep_hours * 3600

    # PDD (Pressure-Dependent Demand) configuration options
    wn.options.hydraulic.demand_model = "PDD"
    wn.options.hydraulic.minimum_pressure = 0.0      # m
    wn.options.hydraulic.required_pressure = 15.0    # m
    wn.options.hydraulic.pressure_exponent = 0.5

    # ---------------------------------------------------------
    # 1. Reservoir (Water Source)
    # ---------------------------------------------------------
    wn.add_reservoir(
        name="R1",
        base_head=60.0,
        coordinates=(0.0, 500.0),
    )

    # ---------------------------------------------------------
    # 2. Diurnal Demand Patterns
    # ---------------------------------------------------------
    # Residential diurnal pattern (peaks at 8 AM and 7 PM)
    residential_multipliers = [
        0.55, 0.45, 0.40, 0.40, 0.50, 0.70,
        1.10, 1.45, 1.40, 1.20, 1.05, 1.00,
        0.95, 0.90, 0.90, 0.95, 1.10, 1.35,
        1.50, 1.40, 1.25, 1.05, 0.80, 0.65,
    ]
    wn.add_pattern("PAT_RESIDENTIAL", residential_multipliers)

    # Commercial diurnal pattern (active during business hours 9 AM - 6 PM)
    commercial_multipliers = [
        0.30, 0.25, 0.25, 0.25, 0.30, 0.40,
        0.60, 0.90, 1.30, 1.50, 1.55, 1.45,
        1.40, 1.40, 1.45, 1.40, 1.30, 1.00,
        0.70, 0.50, 0.40, 0.35, 0.30, 0.30,
    ]
    wn.add_pattern("PAT_COMMERCIAL", commercial_multipliers)

    # ---------------------------------------------------------
    # 3. Junctions (15 Junctions across Zones A, B, and C)
    # ---------------------------------------------------------
    # Zone A (Industrial & Extraction District)
    zone_a_junctions = [
        ("J_A1", 0.008, 15.0, (200.0, 600.0), "PAT_COMMERCIAL"),
        ("J_A2", 0.009, 15.5, (400.0, 650.0), "PAT_COMMERCIAL"),
        ("J_A3", 0.010, 16.0, (600.0, 650.0), "PAT_COMMERCIAL"),
        ("J_A4", 0.007, 15.0, (600.0, 500.0), "PAT_COMMERCIAL"),
        ("J_A5", 0.008, 14.5, (400.0, 480.0), "PAT_COMMERCIAL"),
    ]

    # Zone B (Commercial & Dense Residential Hub - Fault Target)
    zone_b_junctions = [
        ("J_B1", 0.012, 17.0, (200.0, 320.0), "PAT_RESIDENTIAL"),
        ("J_B2", 0.014, 18.0, (400.0, 320.0), "PAT_RESIDENTIAL"),
        ("J_B3", 0.011, 18.5, (600.0, 320.0), "PAT_RESIDENTIAL"),
        ("J_B4", 0.013, 19.0, (600.0, 200.0), "PAT_RESIDENTIAL"),
        ("J_B5", 0.010, 17.5, (400.0, 200.0), "PAT_RESIDENTIAL"),
    ]

    # Zone C (Elevated Suburban District)
    zone_c_junctions = [
        ("J_C1", 0.008, 20.0, (200.0, 50.0), "PAT_RESIDENTIAL"),
        ("J_C2", 0.009, 21.0, (400.0, 50.0), "PAT_RESIDENTIAL"),
        ("J_C3", 0.007, 22.0, (600.0, 50.0), "PAT_RESIDENTIAL"),
        ("J_C4", 0.010, 21.5, (600.0, -80.0), "PAT_RESIDENTIAL"),
        ("J_C5", 0.008, 20.5, (400.0, -80.0), "PAT_RESIDENTIAL"),
    ]

    all_zone_configs = [
        ("Zone_A", zone_a_junctions),
        ("Zone_B", zone_b_junctions),
        ("Zone_C", zone_c_junctions),
    ]

    for zone_name, j_list in all_zone_configs:
        for name, demand, elev, coords, pat in j_list:
            wn.add_junction(
                name=name,
                base_demand=demand,
                demand_pattern=pat,
                elevation=elev,
                coordinates=coords,
            )
            node = wn.get_node(name)
            node.zone = zone_name
            node.tag = zone_name

    # ---------------------------------------------------------
    # 4. Pump Curves & Pumps (2 Supply Components)
    # ---------------------------------------------------------
    # Primary lift pump curve (lifts from Reservoir R1 to Zone A)
    wn.add_curve(
        "PC_MAIN",
        "HEAD",
        [(0.0, 55.0), (0.08, 50.0), (0.16, 42.0), (0.24, 30.0), (0.30, 15.0)],
    )
    wn.add_pump(
        name="PUMP_1",
        start_node_name="R1",
        end_node_name="J_A1",
        pump_type="HEAD",
        pump_parameter="PC_MAIN",
        initial_status="OPEN",
    )

    # Secondary booster pump curve (boosts flow from Zone A to Zone C)
    wn.add_curve(
        "PC_BOOST",
        "HEAD",
        [(0.0, 35.0), (0.05, 30.0), (0.10, 22.0), (0.15, 12.0)],
    )
    wn.add_pump(
        name="PUMP_2",
        start_node_name="J_A3",
        end_node_name="J_C1",
        pump_type="HEAD",
        pump_parameter="PC_BOOST",
        initial_status="OPEN",
    )

    # ---------------------------------------------------------
    # 5. Isolation Valves (3 Valves)
    # ---------------------------------------------------------
    # VALVE_ISO_A: Intra-zone isolation inside Zone A (between J_A1 and J_A2)
    wn.add_valve(
        name="VALVE_ISO_A",
        start_node_name="J_A1",
        end_node_name="J_A2",
        diameter=0.30,
        valve_type="TCV",
        initial_setting=0.0,
        initial_status="OPEN",
    )

    # VALVE_ISO_B: Inter-zone feeder isolation valve (Zone A -> Zone B)
    wn.add_valve(
        name="VALVE_ISO_B",
        start_node_name="J_A5",
        end_node_name="J_B1",
        diameter=0.30,
        valve_type="TCV",
        initial_setting=0.0,
        initial_status="OPEN",
    )

    # VALVE_ISO_C: Inter-zone isolation valve (Zone B -> Zone C)
    wn.add_valve(
        name="VALVE_ISO_C",
        start_node_name="J_B4",
        end_node_name="J_C4",
        diameter=0.30,
        valve_type="TCV",
        initial_setting=0.0,
        initial_status="OPEN",
    )

    # ---------------------------------------------------------
    # 6. Pipes (17 Pipes across 3 Zones)
    # ---------------------------------------------------------
    pipes_spec = [
        # Zone A Transmission & Ring Pipes
        ("P_A1", "J_A2", "J_A3", 200.0, 0.25, 120.0),
        ("P_A2", "J_A3", "J_A4", 200.0, 0.25, 120.0),
        ("P_A3", "J_A4", "J_A5", 200.0, 0.20, 120.0),
        ("P_A4", "J_A5", "J_A2", 400.0, 0.20, 120.0),
        ("P_A5", "J_A1", "J_A4", 450.0, 0.20, 120.0),

        # Zone B Commercial Ring & Cross Pipes
        ("P_B1", "J_B1", "J_B2", 200.0, 0.25, 120.0),
        ("P_B2", "J_B2", "J_B3", 200.0, 0.20, 120.0),
        ("P_B3", "J_B3", "J_B4", 200.0, 0.20, 120.0),
        ("P_B4", "J_B4", "J_B5", 200.0, 0.20, 120.0),
        ("P_B5", "J_B5", "J_B1", 400.0, 0.20, 120.0),
        ("P_B6", "J_B2", "J_B4", 350.0, 0.20, 120.0),

        # Zone C Residential Ring & Cross Pipes
        ("P_C1", "J_C1", "J_C2", 200.0, 0.25, 120.0),
        ("P_C2", "J_C2", "J_C3", 200.0, 0.20, 120.0),
        ("P_C3", "J_C3", "J_C4", 200.0, 0.20, 120.0),
        ("P_C4", "J_C4", "J_C5", 200.0, 0.20, 120.0),
        ("P_C5", "J_C5", "J_C1", 400.0, 0.20, 120.0),
        ("P_C6", "J_C2", "J_C5", 350.0, 0.20, 120.0),
    ]

    for p_name, start_n, end_n, length, diameter, roughness in pipes_spec:
        wn.add_pipe(
            name=p_name,
            start_node_name=start_n,
            end_node_name=end_n,
            length=length,
            diameter=diameter,
            roughness=roughness,
            initial_status="OPEN",
        )

    return wn


def get_nodes_by_zone(wn: wntr.network.WaterNetworkModel, zone_name: str) -> List[str]:
    """
    Returns the list of junction names belonging to a specific zone.
    """
    normalized_zone = zone_name.replace(" ", "_")
    matched = []
    for j_name in wn.junction_name_list:
        node = wn.get_node(j_name)
        zone = getattr(node, "zone", "")
        if zone == normalized_zone or zone == zone_name or getattr(node, "tag", "") == zone_name:
            matched.append(j_name)
    return matched


def get_network_summary(wn: wntr.network.WaterNetworkModel) -> Dict[str, Any]:
    """
    Returns structural and inventory summary of the water network model.
    """
    zones: Dict[str, List[str]] = {}
    for j_name in wn.junction_name_list:
        node = wn.get_node(j_name)
        z = getattr(node, "zone", "Unassigned")
        zones.setdefault(z, []).append(j_name)

    return {
        "num_reservoirs": wn.num_reservoirs,
        "num_tanks": wn.num_tanks,
        "num_junctions": wn.num_junctions,
        "num_pipes": wn.num_pipes,
        "num_pumps": wn.num_pumps,
        "num_valves": wn.num_valves,
        "num_links": wn.num_links,
        "num_nodes": wn.num_nodes,
        "zones": {k: len(v) for k, v in zones.items()},
        "total_base_demand_lps": sum(
            j.base_demand * 1000.0 for _, j in wn.junctions()
        ),
    }


if __name__ == "__main__":
    model = build_network()
    summary = get_network_summary(model)
    print("Network Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
