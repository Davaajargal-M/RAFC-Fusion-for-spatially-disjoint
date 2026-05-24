from __future__ import annotations
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

class FusionMode(Enum):
    """Available fusion operators."""

    BASELINE = "baseline"
    BASELINE_GHOST = "baseline_ghost"
    GEOMETRY_AWARE = "geometry_aware"
    ADAPTIVE_GATES = "adaptive_gates"
    CROSS_MODAL_TRANSFORMER = "cross_modal_transformer"
    PHYSICS_INFORMED = "physics_informed"


@dataclass
class ControlSignal:
    """Output of the PID feedback controller."""

    action: str
    magnitude: float
    error: float
    timestamp: float


@dataclass
class DataCharacteristics:
    """Training-data characterization signals used for feedforward scoring."""

    terrain_complexity: float
    spectral_diversity: float
    shadow_ratio: float
    spatial_correlation: float
    modality_agreement: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "terrain_complexity": float(self.terrain_complexity),
            "spectral_diversity": float(self.spectral_diversity),
            "shadow_ratio": float(self.shadow_ratio),
            "spatial_correlation": float(self.spatial_correlation),
            "modality_agreement": float(self.modality_agreement),
        }


class PIDController:
    """PID feedback controller that maps validation accuracy error to control actions."""

    def __init__(
        self,
        target: float = 0.95,
        kp: float = 0.4,
        ki: float = 0.05,
        kd: float = 0.3,
        integral_limit: float = 1.0,
    ) -> None:
        self.target = float(target)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.integral_limit = float(integral_limit)

        self.error_integral = 0.0
        self.last_error = 0.0
        self.history: List[Dict[str, float | str]] = []

    def compute(self, current_value: float) -> ControlSignal:
        error = self.target - float(current_value)

        p_term = self.kp * error
        self.error_integral = float(
            np.clip(self.error_integral + error, -self.integral_limit, self.integral_limit)
        )
        i_term = self.ki * self.error_integral
        d_term = self.kd * (error - self.last_error)
        self.last_error = error

        control_value = p_term + i_term + d_term

        if control_value > 0.15:
            action = "enable_complex"
        elif control_value > 0.05:
            action = "enable_simple"
        elif control_value < -0.05:
            action = "disable"
        else:
            action = "maintain"

        signal = ControlSignal(
            action=action,
            magnitude=float(abs(control_value)),
            error=float(error),
            timestamp=time.time(),
        )

        self.history.append(
            {
                "error": float(error),
                "p": float(p_term),
                "i": float(i_term),
                "d": float(d_term),
                "control": float(control_value),
                "action": action,
            }
        )
        return signal

    def set_target(self, target: float) -> None:
        self.target = float(target)

    def reset(self) -> None:
        self.error_integral = 0.0
        self.last_error = 0.0
        self.history.clear()


class DatasetCharacterizer:
    """Computes train-only characterization signals and feedforward operator scores."""

    _KEYS = (
        "terrain_complexity",
        "spectral_diversity",
        "shadow_ratio",
        "spatial_correlation",
        "modality_agreement",
    )

    def __init__(self) -> None:
        self.feature_stats: Optional[Dict[str, Dict[str, float]]] = None

    @staticmethod
    def _sigmoid(x: float) -> float:
        x = float(np.clip(x, -20.0, 20.0))
        return 1.0 / (1.0 + np.exp(-x))

    def fit_statistics(self, characteristics_list: List[DataCharacteristics]) -> None:
        if not characteristics_list:
            raise ValueError("characteristics_list must not be empty.")

        self.feature_stats = {}
        for key in self._KEYS:
            values = np.asarray([getattr(item, key) for item in characteristics_list], dtype=np.float32)
            self.feature_stats[key] = {
                "mean": float(values.mean()),
                "std": float(values.std() + 1e-6),
            }

    def analyze(self, hsi_data: torch.Tensor, lidar_data: torch.Tensor) -> DataCharacteristics:
        with torch.no_grad():
            return DataCharacteristics(
                terrain_complexity=self._compute_terrain_complexity(lidar_data),
                spectral_diversity=self._compute_spectral_diversity(hsi_data),
                shadow_ratio=self._compute_shadow_ratio(hsi_data, lidar_data),
                spatial_correlation=self._compute_spatial_correlation(hsi_data),
                modality_agreement=self._compute_modality_agreement(hsi_data, lidar_data),
            )

    def recommend_scores(self, characteristics: DataCharacteristics) -> Dict[FusionMode, float]:
        if self.feature_stats is None:
            raise RuntimeError("DatasetCharacterizer.fit_statistics must be called before recommend_scores.")

        def z_score(value: float, key: str) -> float:
            stats = self.feature_stats[key]
            return (float(value) - stats["mean"]) / stats["std"]

        tc = self._sigmoid(z_score(characteristics.terrain_complexity, "terrain_complexity"))
        sd = self._sigmoid(z_score(characteristics.spectral_diversity, "spectral_diversity"))
        sr = self._sigmoid(z_score(characteristics.shadow_ratio, "shadow_ratio"))
        sc = self._sigmoid(-z_score(characteristics.spatial_correlation, "spatial_correlation"))
        ma = self._sigmoid(-z_score(characteristics.modality_agreement, "modality_agreement"))

        scores = {
            FusionMode.BASELINE: 0.10,
            FusionMode.BASELINE_GHOST: 1.0 - np.mean([tc, sd, sr, ma]),
            FusionMode.GEOMETRY_AWARE: np.mean([tc, ma, sr]),
            FusionMode.ADAPTIVE_GATES: ma,
            FusionMode.CROSS_MODAL_TRANSFORMER: np.mean([sd, sc]),
            FusionMode.PHYSICS_INFORMED: sr,
        }
        return {mode: float(np.clip(score, 0.05, 0.95)) for mode, score in scores.items()}

    @staticmethod
    def _compute_terrain_complexity(lidar: torch.Tensor) -> float:
        elevation_std = lidar.std().item()
        dx = lidar[:, :, 1:, :] - lidar[:, :, :-1, :]
        dy = lidar[:, :, :, 1:] - lidar[:, :, :, :-1]
        gradient = torch.sqrt(dx[:, :, :, :-1].pow(2) + dy[:, :, :-1, :].pow(2))
        complexity = gradient.std().item() / (elevation_std + 1e-6)
        return float(np.clip(complexity, 0.0, 1.0))

    @staticmethod
    def _compute_spectral_diversity(hsi: torch.Tensor) -> float:
        spectral_means = hsi.mean(dim=[0, 2, 3])
        spectral_stds = hsi.std(dim=[0, 2, 3])
        coefficient_variation = (spectral_stds / (spectral_means + 1e-6)).mean().item()
        return float(np.clip(coefficient_variation, 0.0, 1.0))

    @staticmethod
    def _compute_shadow_ratio(hsi: torch.Tensor, lidar: torch.Tensor) -> float:
        brightness = hsi.mean(dim=1, keepdim=True)
        dx = lidar[:, :, 1:, :] - lidar[:, :, :-1, :]
        dy = lidar[:, :, :, 1:] - lidar[:, :, :, :-1]
        slope = torch.sqrt(dx[:, :, :, :-1].pow(2) + dy[:, :, :-1, :].pow(2))
        slope = F.pad(slope, (0, 1, 0, 1), mode="replicate")
        brightness = brightness[:, :, : slope.shape[2], : slope.shape[3]]
        shadow = (brightness < brightness.mean()) & (slope > slope.mean())
        return float(np.clip(shadow.float().mean().item(), 0.0, 1.0))

    @staticmethod
    def _compute_spatial_correlation(hsi: torch.Tensor) -> float:
        hsi_flat = hsi.reshape(hsi.shape[0], hsi.shape[1], -1)
        centered = hsi_flat - hsi_flat.mean(dim=2, keepdim=True)
        shifted = torch.roll(centered, shifts=1, dims=2)
        correlation = (centered * shifted).mean().item()
        variance = centered.var().item()
        return float(np.clip(correlation / (variance + 1e-6), 0.0, 1.0))

    @staticmethod
    def _compute_modality_agreement(hsi: torch.Tensor, lidar: torch.Tensor) -> float:
        hsi_feature = hsi.mean(dim=1, keepdim=True)
        lidar_feature = lidar.mean(dim=1, keepdim=True)

        if hsi_feature.shape[2:] != lidar_feature.shape[2:]:
            lidar_feature = F.interpolate(
                lidar_feature,
                size=hsi_feature.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        hsi_flat = hsi_feature.flatten()
        lidar_flat = lidar_feature.flatten()
        hsi_centered = hsi_flat - hsi_flat.mean()
        lidar_centered = lidar_flat - lidar_flat.mean()
        denom = torch.sqrt(hsi_centered.pow(2).sum() * lidar_centered.pow(2).sum()) + 1e-6
        correlation = (hsi_centered * lidar_centered).sum() / denom
        return float(torch.clamp((correlation + 1.0) / 2.0, 0.0, 1.0).item())


class AdaptiveTargetController:
    """Sets a reachable validation target from the baseline validation accuracy."""

    def __init__(self, initial_target: float = 0.95) -> None:
        self.initial_target = float(initial_target)
        self.current_target = float(initial_target)

    def adjust_target(
        self,
        dataset_name: str,
        baseline_accuracy: float,
        characteristics: Optional[DataCharacteristics] = None,
    ) -> float:
        del dataset_name
        baseline_accuracy = float(baseline_accuracy)

        if baseline_accuracy < 0.85:
            target = min(baseline_accuracy + 0.10, 0.98)
        elif baseline_accuracy > 0.95:
            target = 0.99
        else:
            target = min(baseline_accuracy + 0.05, 0.97)

        if characteristics is not None:
            if characteristics.terrain_complexity > 0.7:
                target -= 0.02
            if characteristics.modality_agreement < 0.5:
                target -= 0.02

        self.current_target = max(target, baseline_accuracy + 0.02)
        return float(self.current_target)


class HybridFusionController:
    """Combines feedforward operator priors with PID-based feedback scores."""

    def __init__(
        self,
        target_accuracy: float = 0.95,
        alpha: float = 0.8,
        beta: float = 0.2,
        epsilon: float = 0.05,
    ) -> None:
        self.pid_controller = PIDController(target=target_accuracy)
        self.characterizer = DatasetCharacterizer()
        self.target_controller = AdaptiveTargetController(initial_target=target_accuracy)

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.epsilon = float(epsilon)

        self.method_memory: Dict[FusionMode, List[float]] = {mode: [] for mode in FusionMode}
        self.selection_history: List[Dict] = []
        self.control_signal_history: List[Dict] = []
        self.operator_score_history: List[Dict[str, float]] = []
        self.reliability_history: List[Dict[str, float]] = []

        self.reliability_lambda = 0.5
        self.reliability_scale = 10.0

    def control(
        self,
        hsi_data: torch.Tensor,
        lidar_data: torch.Tensor,
        current_accuracy: float,
        dataset_name: Optional[str] = None,
    ) -> Tuple[List[FusionMode], DataCharacteristics, ControlSignal]:
        del dataset_name
        characteristics = self.characterizer.analyze(hsi_data, lidar_data)
        feedforward_scores = self.characterizer.recommend_scores(characteristics)
        control_signal = self.pid_controller.compute(current_accuracy)

        self.control_signal_history.append(
            {
                "error": float(control_signal.error),
                "magnitude": float(control_signal.magnitude),
                "action": control_signal.action,
            }
        )

        selected, feedback_scores, combined_scores = self._combine_signals(feedforward_scores, control_signal)
        self.selection_history.append(
            {
                "ff_scores": {mode.value: float(score) for mode, score in feedforward_scores.items()},
                "fb_scores": {mode.value: float(score) for mode, score in feedback_scores.items()},
                "combined_scores": {mode.value: float(score) for mode, score in combined_scores.items()},
                "control_action": control_signal.action,
                "selected": [mode.value for mode in selected],
            }
        )

        return selected, characteristics, control_signal

    def _combine_signals(
        self,
        ff_scores: Dict[FusionMode, float],
        control_signal: ControlSignal,
    ) -> Tuple[List[FusionMode], Dict[FusionMode, float], Dict[FusionMode, float]]:
        all_operators = list(FusionMode)
        action_gain = {
            "enable_complex": 1.0,
            "enable_simple": 0.7,
            "maintain": 0.4,
            "disable": 0.1,
        }.get(control_signal.action, 0.4)

        reliability_scores = {mode: self.compute_operator_reliability(mode) for mode in all_operators}
        feedback_scores = {mode: reliability_scores[mode] * action_gain for mode in all_operators}

        total_feedback = sum(feedback_scores.values()) + 1e-8
        feedback_scores = {mode: score / total_feedback for mode, score in feedback_scores.items()}

        combined_scores = {
            mode: self.alpha * ff_scores.get(mode, 0.0) + self.beta * feedback_scores.get(mode, 0.0)
            for mode in all_operators
        }

        self.reliability_history.append({mode.value: float(reliability_scores[mode]) for mode in all_operators})
        self.operator_score_history.append({mode.value: float(combined_scores[mode]) for mode in all_operators})

        ranked = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
        selected = [ranked[0][0]]
        if len(ranked) > 1:
            if random.random() < self.epsilon:
                selected.append(random.choice([mode for mode, _ in ranked[1:]]))
            else:
                selected.append(ranked[1][0])

        return list(dict.fromkeys(selected)), feedback_scores, combined_scores

    def compute_operator_reliability(self, method: FusionMode) -> float:
        history = self.method_memory.get(method, [])
        if len(history) < 2:
            return 0.5

        values = np.asarray(history, dtype=np.float32)
        trend = values[-1] - values[:-1].mean()
        stability = values.std()
        score = trend - self.reliability_lambda * stability
        return float(DatasetCharacterizer._sigmoid(score * self.reliability_scale))

    def set_target(self, target: float) -> None:
        self.pid_controller.set_target(target)
        self.target_controller.current_target = float(target)

    def reset(self) -> None:
        self.pid_controller.reset()
        self.method_memory = {mode: [] for mode in FusionMode}
        self.selection_history.clear()
        self.control_signal_history.clear()
        self.operator_score_history.clear()
        self.reliability_history.clear()
