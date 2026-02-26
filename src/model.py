import json
import joblib
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class ThresholdConfig:
    """
    Stores anomaly threshold settings.
    strategy: "p95" or "z" or "severity"
    params: dictionary of parameters needed for that strategy
    """
    strategy: str
    params: Dict[str, Any]

def load_model(model_path: str):
    return joblib.load(model_path)

def save_threshold_config(cfg: ThresholdConfig, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"strategy": cfg.strategy, "params": cfg.params}, f, indent=2)

def load_threshold_config(path: str) -> ThresholdConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ThresholdConfig(strategy=data["strategy"], params=data["params"])