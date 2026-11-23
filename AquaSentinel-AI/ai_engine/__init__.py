"""
AquaSentinel AI - AI Engine for Water Distribution Networks
Provides Anomaly Detection, Fault Classification, NetworkX Topological
Localization, and Physics-Based Sensor Fault Diagnosis.
"""

from .data_generator import generate_dataset, ScenarioDataset
from .anomaly_detection import HydraulicAnomalyDetector
from .fault_classifier import FaultClassifier
from .fault_localisation import FaultLocaliser
from .sensor_fault import SensorFaultDiagnoser, SensorFaultInjector

__all__ = [
    "generate_dataset",
    "ScenarioDataset",
    "HydraulicAnomalyDetector",
    "FaultClassifier",
    "FaultLocaliser",
    "SensorFaultDiagnoser",
    "SensorFaultInjector",
]
