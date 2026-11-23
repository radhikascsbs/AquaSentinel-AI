"""
AquaSentinel AI - Scenario Dataset Generator
Executes real WNTR hydraulic simulations across diverse operating conditions,
demand distributions, and physical anomalies to produce labeled training & test datasets.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from simulation.network_builder import build_network
from simulation.hydraulic_engine import HydraulicEngine
from simulation.fault_injection import FaultInjector


@dataclass
class ScenarioDataset:
    """Encapsulates features, labels, metadata, and train/test splits."""
    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    meta_train: pd.DataFrame
    meta_test: pd.DataFrame
    feature_names: List[str]


def generate_dataset(
    n_samples: int = 120,
    leak_ratio: float = 0.5,
    test_size: float = 0.25,
    demand_variation_range: Tuple[float, float] = (0.80, 1.25),
    leak_area_range: Tuple[float, float] = (0.0008, 0.0040),
    pressure_noise_std: float = 0.05,
    flow_noise_std: float = 0.0001,
    random_state: int = 42,
    save_dir: Optional[str] = "data/ai_datasets",
) -> ScenarioDataset:
    """
    Generates a balanced dataset of normal and leak scenarios using real WNTR simulations.

    Parameters
    ----------
    n_samples : int
        Total number of simulation scenarios to generate.
    leak_ratio : float
        Fraction of scenarios that contain physical leaks (default 0.5 = balanced).
    test_size : float
        Fraction of data reserved for the test set (default 0.25).
    demand_variation_range : tuple
        Min and max multiplier for junction base demand per scenario.
    leak_area_range : tuple
        Min and max orifice area in m^2 for injected leaks.
    pressure_noise_std : float
        Gaussian sensor noise standard deviation added to pressure (m).
    flow_noise_std : float
        Gaussian sensor noise standard deviation added to flowrate (m^3/s).
    random_state : int
        RNG seed for reproducibility.
    save_dir : str, optional
        Path to save CSV datasets.

    Returns
    -------
    ScenarioDataset
        Dataset container with features, labels, and train/test splits.
    """
    rng = np.random.RandomState(random_state)
    n_leak = int(n_samples * leak_ratio)
    n_normal = n_samples - n_leak

    # Candidate leak nodes across all 3 zones (interior junctions)
    candidate_leak_nodes = [
        "J_A2", "J_A3", "J_A4", "J_A5",
        "J_B1", "J_B2", "J_B3", "J_B4", "J_B5",
        "J_C2", "J_C3", "J_C4", "J_C5",
    ]

    scenario_records: List[Dict[str, float]] = []
    meta_records: List[Dict[str, Any]] = []

    # Prepare scenario definitions
    scenarios_plan = [("normal", None, 0.0)] * n_normal
    for _ in range(n_leak):
        target_junc = rng.choice(candidate_leak_nodes)
        area = float(rng.uniform(leak_area_range[0], leak_area_range[1]))
        scenarios_plan.append(("leak", target_junc, area))

    rng.shuffle(scenarios_plan)

    for idx, (label_str, target_node, leak_area) in enumerate(scenarios_plan):
        # 1. Build fresh network model
        wn = build_network(duration_hours=0)  # Snapshot simulation

        # 2. Apply random demand variations across all junctions
        demand_mult = float(rng.uniform(demand_variation_range[0], demand_variation_range[1]))
        for _, junction in wn.junctions():
            # Perturb each junction base demand independently around the scenario multiplier
            local_perturb = rng.uniform(0.95, 1.05)
            junction.demand_timeseries_list[0].base_value *= (demand_mult * local_perturb)

        # 3. Inject leak if scenario is a leak
        leak_discharge_actual = 0.0
        leak_zone = "None"
        if label_str == "leak" and target_node is not None:
            node = wn.get_node(target_node)
            leak_zone = getattr(node, "zone", "Unknown")
            FaultInjector.inject_junction_leak(
                wn=wn,
                junction_name=target_node,
                leak_area=leak_area,
                discharge_coeff=0.75,
                start_time=0,
            )

        # 4. Run hydraulic simulation with WNTR
        engine = HydraulicEngine(wn)
        results = engine.run(engine="WNTR", mode="PDD")

        # 5. Extract snapshot pressure and flowrate measurements
        p_snapshot = results.pressure.iloc[0].to_dict()
        q_snapshot = results.flowrate.iloc[0].to_dict()

        if label_str == "leak" and target_node in results.leak_demand.columns:
            leak_discharge_actual = float(results.leak_demand[target_node].iloc[0]) * 1000.0  # L/s

        # 6. Formulate feature vector with realistic sensor noise
        feature_dict: Dict[str, float] = {}

        # Pressure sensors on all 15 junctions
        for j_name in sorted(wn.junction_name_list):
            raw_p = p_snapshot.get(j_name, 0.0)
            noise_p = rng.normal(0.0, pressure_noise_std)
            feature_dict[f"P_{j_name}"] = float(raw_p + noise_p)

        # Flow sensors on pipes, pumps, valves
        for l_name in sorted(wn.link_name_list):
            raw_q = q_snapshot.get(l_name, 0.0) * 1000.0  # L/s
            noise_q = rng.normal(0.0, flow_noise_std * 1000.0)
            feature_dict[f"Q_{l_name}"] = float(raw_q + noise_q)

        scenario_records.append(feature_dict)

        meta_records.append({
            "scenario_id": idx,
            "label": label_str,
            "is_leak": 1 if label_str == "leak" else 0,
            "target_node": target_node if target_node else "None",
            "zone": leak_zone,
            "leak_area_m2": leak_area,
            "leak_flow_lps": leak_discharge_actual,
            "scenario_demand_mult": demand_mult,
        })

    X = pd.DataFrame(scenario_records)
    meta_df = pd.DataFrame(meta_records)
    y = meta_df["is_leak"]

    # Stratified Train/Test split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        meta_df.index,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    meta_train = meta_df.loc[idx_train].reset_index(drop=True)
    meta_test = meta_df.loc[idx_test].reset_index(drop=True)

    # Save to disk if requested
    if save_dir:
        out_p = Path(save_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        X_train.to_csv(out_p / "X_train.csv", index=False)
        X_test.to_csv(out_p / "X_test.csv", index=False)
        y_train.to_csv(out_p / "y_train.csv", index=False)
        y_test.to_csv(out_p / "y_test.csv", index=False)
        meta_train.to_csv(out_p / "meta_train.csv", index=False)
        meta_test.to_csv(out_p / "meta_test.csv", index=False)

    return ScenarioDataset(
        X=X,
        y=y,
        metadata=meta_df,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        meta_train=meta_train,
        meta_test=meta_test,
        feature_names=list(X.columns),
    )


if __name__ == "__main__":
    print("Generating sample hydraulic dataset...")
    dataset = generate_dataset(n_samples=20, test_size=0.25)
    print(f"Generated {len(dataset.X)} samples.")
    print(f"Features ({len(dataset.feature_names)}): {dataset.feature_names[:6]} ...")
    print(f"Train samples: {len(dataset.X_train)}, Test samples: {len(dataset.X_test)}")
