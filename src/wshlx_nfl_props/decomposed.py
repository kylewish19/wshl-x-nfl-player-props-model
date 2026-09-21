from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class DecomposedYardageModel:
    prop: str
    kind: str
    estimator_name: str
    feature_set: str
    volume_pipeline: object
    efficiency_pipeline: object
    numeric_features: list[str]
    categorical_features: list[str]
    volume_target: str
    yardage_target: str
    efficiency_clip: tuple[float, float]
    residuals: list[float]
    validation_metrics: dict

    def predict_components(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        volume = np.asarray(self.volume_pipeline.predict(X), dtype=float)
        efficiency = np.asarray(self.efficiency_pipeline.predict(X), dtype=float)
        volume = np.clip(volume, 0.0, None)
        efficiency = np.clip(efficiency, self.efficiency_clip[0], self.efficiency_clip[1])
        return volume, efficiency

    def predict_point(self, X: pd.DataFrame) -> np.ndarray:
        volume, efficiency = self.predict_components(X)
        return volume * efficiency

    def probability_over(self, X: pd.DataFrame, line: float) -> np.ndarray:
        point = self.predict_point(X)
        residuals = np.asarray(self.residuals, dtype=float)
        if residuals.size < 20:
            residuals = np.array([0.0])
        return np.array([(p + residuals > line).mean() for p in point])
