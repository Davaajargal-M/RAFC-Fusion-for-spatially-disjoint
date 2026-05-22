from __future__ import annotations
import copy
import logging
import time
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from feedback_fusion_control import (
    AdaptiveTargetController,
    DatasetCharacterizer,
    FusionMode,
    HybridFusionController,
)

LOGGER = logging.getLogger(__name__)


class LayerNorm2d(nn.Module):
    """LayerNorm for channel-first tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class GhostConv(nn.Module):
    """Lightweight GhostConv block: primary convolution plus cheap expansion."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, ratio: int = 2) -> None:
        super().__init__()
        primary_channels = max(out_channels // ratio, 1)
        cheap_channels = out_channels - primary_channels
        padding = kernel_size // 2

        self.primary = nn.Conv2d(in_channels, primary_channels, kernel_size, padding=padding, bias=False)
        self.cheap = nn.Conv2d(primary_channels, cheap_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        primary = self.primary(x)
        cheap = self.cheap(primary)
        return self.act(self.norm(torch.cat([primary, cheap], dim=1)))


class BaselineFusion(nn.Module):
    """Baseline concatenation fusion."""

    def __init__(self, hsi_channels: int, lidar_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.proj_hsi = nn.Conv2d(hsi_channels, embed_dim, kernel_size=1)
        self.proj_lidar = nn.Conv2d(lidar_channels, embed_dim, kernel_size=1)
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, hsi: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        hsi_feat = self.proj_hsi(hsi)
        lidar_feat = self.proj_lidar(lidar)
        fused = self.fusion(torch.cat([hsi_feat, lidar_feat], dim=1))
        return self.pool(fused).flatten(1)


class BaselineGhostFusion(nn.Module):
    """GhostConv-based baseline fusion."""

    def __init__(self, hsi_channels: int, lidar_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.hsi_ghost = GhostConv(hsi_channels, embed_dim)
        self.lidar_ghost = GhostConv(lidar_channels, embed_dim)
        self.fusion = nn.Sequential(nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1), nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, hsi: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        hsi_feat = self.hsi_ghost(hsi)
        lidar_feat = self.lidar_ghost(lidar)
        fused = self.fusion(torch.cat([hsi_feat, lidar_feat], dim=1))
        return self.pool(fused).flatten(1)


class GeometryAwareGhostFusion(nn.Module):
    """LiDAR-guided residual geometry refinement."""

    def __init__(self, hsi_channels: int, lidar_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.hsi_ghost = GhostConv(hsi_channels, embed_dim)
        self.lidar_ghost = GhostConv(lidar_channels, embed_dim)
        self.baseline_fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1),
            LayerNorm2d(embed_dim),
            nn.GELU(),
        )
        self.geometry_extractor = nn.Sequential(
            nn.Conv2d(lidar_channels, embed_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.geometry_refine = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            LayerNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, hsi: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        hsi_feat = self.hsi_ghost(hsi)
        lidar_feat = self.lidar_ghost(lidar)
        baseline = self.baseline_fusion(torch.cat([hsi_feat, lidar_feat], dim=1))

        geo_gate = self.geometry_extractor(lidar)
        geo_correction = self.geometry_refine(baseline * geo_gate)
        geo_confidence = geo_gate.mean(dim=(1, 2, 3), keepdim=True)

        fused = baseline + 0.1 * geo_confidence * geo_correction
        return self.pool(fused).flatten(1)


class AdaptiveGateGhostFusion(nn.Module):
    """Cosine-prior adaptive gating fusion."""

    def __init__(self, hsi_channels: int, lidar_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.hsi_ghost = GhostConv(hsi_channels, embed_dim)
        self.lidar_ghost = GhostConv(lidar_channels, embed_dim)
        self.gate_network = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, 2, kernel_size=1),
            nn.Softmax(dim=1),
        )
        self.fusion = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, kernel_size=1), nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, hsi: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        hsi_feat = self.hsi_ghost(hsi)
        lidar_feat = self.lidar_ghost(lidar)

        combined = torch.cat([hsi_feat, lidar_feat], dim=1)
        learned_gate = self.gate_network(combined)[:, 0:1]

        cos_sim = F.cosine_similarity(
            F.normalize(hsi_feat, dim=1),
            F.normalize(lidar_feat, dim=1),
            dim=1,
        ).unsqueeze(1)
        cos_gate = torch.clamp((cos_sim + 1.0) / 2.0, 0.0, 1.0)

        gate_hsi = 0.7 * cos_gate + 0.3 * learned_gate
        fused = gate_hsi * hsi_feat + (1.0 - gate_hsi) * lidar_feat
        return self.pool(self.fusion(fused)).flatten(1)


class CrossModalTransformerFusion(nn.Module):
    """Bidirectional cross-attention fusion."""

    def __init__(self, hsi_channels: int, lidar_channels: int, embed_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.hsi_ghost = GhostConv(hsi_channels, embed_dim)
        self.lidar_ghost = GhostConv(lidar_channels, embed_dim)
        self.hsi_to_lidar = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)
        self.lidar_to_hsi = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.ReLU(inplace=True), nn.Linear(embed_dim, embed_dim))

    def forward(self, hsi: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        hsi_feat = self.hsi_ghost(hsi)
        lidar_feat = self.lidar_ghost(lidar)

        batch, channels, _, _ = hsi_feat.shape
        hsi_seq = hsi_feat.flatten(2).permute(0, 2, 1)
        lidar_seq = lidar_feat.flatten(2).permute(0, 2, 1)

        hsi_attended, _ = self.hsi_to_lidar(hsi_seq, lidar_seq, lidar_seq)
        lidar_attended, _ = self.lidar_to_hsi(lidar_seq, hsi_seq, hsi_seq)

        hsi_pooled = hsi_attended.mean(dim=1)
        lidar_pooled = lidar_attended.mean(dim=1)
        return self.fusion(torch.cat([hsi_pooled, lidar_pooled], dim=1))


class PhysicsInformedGhostFusion(nn.Module):
    """LiDAR-guided shadow correction followed by GhostConv fusion."""

    def __init__(self, hsi_channels: int, lidar_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.geometry_processor = nn.Sequential(
            nn.Conv2d(lidar_channels, embed_dim // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 4, embed_dim // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.shadow_corrector = nn.Sequential(
            nn.Conv2d(hsi_channels + embed_dim // 4, hsi_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hsi_channels, hsi_channels, kernel_size=3, padding=1),
        )
        self.hsi_ghost = GhostConv(hsi_channels, embed_dim)
        self.lidar_ghost = GhostConv(lidar_channels, embed_dim)
        self.fusion = nn.Sequential(nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1), nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, hsi: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        geometry = self.geometry_processor(lidar)
        correction = self.shadow_corrector(torch.cat([hsi, geometry], dim=1))
        hsi_corrected = hsi + correction

        hsi_feat = self.hsi_ghost(hsi_corrected)
        lidar_feat = self.lidar_ghost(lidar)
        fused = self.fusion(torch.cat([hsi_feat, lidar_feat], dim=1))
        return self.pool(fused).flatten(1)


class ControlledFusionModel(nn.Module):
    """Classifier with selectable HSI-LiDAR fusion operators."""

    def __init__(self, hsi_channels: int, lidar_channels: int, num_classes: int, embed_dim: int = 256) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_classes = int(num_classes)

        self.fusion_dict = nn.ModuleDict(
            {
                FusionMode.BASELINE.value: BaselineFusion(hsi_channels, lidar_channels, embed_dim),
                FusionMode.BASELINE_GHOST.value: BaselineGhostFusion(hsi_channels, lidar_channels, embed_dim),
                FusionMode.GEOMETRY_AWARE.value: GeometryAwareGhostFusion(hsi_channels, lidar_channels, embed_dim),
                FusionMode.ADAPTIVE_GATES.value: AdaptiveGateGhostFusion(hsi_channels, lidar_channels, embed_dim),
                FusionMode.CROSS_MODAL_TRANSFORMER.value: CrossModalTransformerFusion(hsi_channels, lidar_channels, embed_dim),
                FusionMode.PHYSICS_INFORMED.value: PhysicsInformedGhostFusion(hsi_channels, lidar_channels, embed_dim),
            }
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(embed_dim // 2, num_classes),
        )

        self.active_mode = FusionMode.BASELINE_GHOST
        self.enable_feedback_refinement = False
        self.refinement_gain = 0.10
        self.ref_hsi_proj = nn.Conv2d(hsi_channels, embed_dim, kernel_size=1)
        self.ref_lidar_proj = nn.Conv2d(lidar_channels, embed_dim, kernel_size=1)
        self.ref_pool = nn.AdaptiveAvgPool2d(1)
        self.refinement_net = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def set_fusion_mode(self, mode: FusionMode) -> None:
        self.active_mode = mode

    def set_feedback_refinement(self, enabled: bool, gain: float = 0.10) -> None:
        self.enable_feedback_refinement = bool(enabled)
        self.refinement_gain = float(gain)

    def forward(self, hsi: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        mode = self.active_mode or FusionMode.BASELINE_GHOST
        fused = self.fusion_dict[mode.value](hsi, lidar)

        if self.enable_feedback_refinement:
            hsi_context = self.ref_pool(self.ref_hsi_proj(hsi)).flatten(1)
            lidar_context = self.ref_pool(self.ref_lidar_proj(lidar)).flatten(1)
            delta = self.refinement_net(torch.cat([fused, hsi_context, lidar_context], dim=1))
            fused = fused + self.refinement_gain * delta

        return self.classifier(fused)


def count_parameters(model: nn.Module) -> float:
    """Return trainable parameters in millions."""

    return sum(param.numel() for param in model.parameters() if param.requires_grad) / 1e6


class ControlledTrainer:
    """Training helper for baseline and feedback-controlled fusion search."""

    def __init__(
        self,
        model: ControlledFusionModel,
        train_loader: Iterable,
        val_loader: Iterable,
        device: str = "cuda",
        lr: float = 1e-4,
        epochs: int = 200,
        weight_decay: float = 5e-4,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.weight_decay = float(weight_decay)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.control_system = HybridFusionController(target_accuracy=0.95)

        self.best_accuracy = 0.0
        self.best_model_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_model_method: Optional[str] = None
        self.best_model_mode: Optional[FusionMode] = None
        self.best_refinement_enabled = False
        self.best_refinement_gain = 0.0

        self.acc_ema: Optional[float] = None
        self.ema_lambda = 0.7
        self.operator_score_memory: Dict[FusionMode, List[float]] = {}
        self.operator_ema_score: Dict[FusionMode, float] = {}
        self.reward_ema_decay = 0.8
        self.stability_lambda = 0.5
        self.confidence_weight = 0.2
        self.method_memory: Dict[FusionMode, List[float]] = {}
        self.method_failure_count: Dict[FusionMode, int] = {}

        self.control_epochs_ratio = 0.1
        self.min_control_epochs = 200
        self.accept_margin = 0.015

    def _new_optimizer(self, lr: Optional[float] = None) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.CosineAnnealingLR]:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr if lr is None else float(lr),
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        return optimizer, scheduler

    def evaluate(self) -> float:
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.val_loader:
                hsi = batch["hsi"].to(self.device)
                lidar = batch["lidar"].to(self.device)
                labels = batch["label"].to(self.device)
                preds = self.model(hsi, lidar).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.numel()
        return correct / max(total, 1)

    def train_with_mode(self, fusion_mode: FusionMode, num_epochs: Optional[int] = None) -> Dict[str, float]:
        num_epochs = self.epochs if num_epochs is None else int(num_epochs)
        self.model.set_fusion_mode(fusion_mode)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        best_acc = 0.0
        best_state = copy.deepcopy(self.model.state_dict())
        val_history: List[float] = []

        for epoch in range(num_epochs):
            self.model.train()
            for batch in self.train_loader:
                hsi = batch["hsi"].to(self.device)
                lidar = batch["lidar"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad(set_to_none=True)
                loss = self.criterion(self.model(hsi, lidar), labels)
                loss.backward()
                optimizer.step()

            scheduler.step()

            if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
                val_acc = self.evaluate()
                val_history.append(float(val_acc))
                if val_acc > best_acc:
                    best_acc = float(val_acc)
                    best_state = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(best_state)
        self.model.set_fusion_mode(fusion_mode)

        return {
            "best_acc": float(best_acc),
            "val_std": float(np.std(val_history[-10:])) if len(val_history) >= 2 else 0.0,
        }

    def _estimate_validation_confidence(self, max_batches: int = 3) -> float:
        self.model.eval()
        confidences: List[float] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx >= max_batches:
                    break
                hsi = batch["hsi"].to(self.device)
                lidar = batch["lidar"].to(self.device)
                probs = torch.softmax(self.model(hsi, lidar), dim=1)
                confidences.append(probs.max(dim=1)[0].mean().item())
        self.model.train()
        return float(np.mean(confidences)) if confidences else 0.0

    def _compute_operator_reward(self, method: FusionMode, val_acc: float) -> float:
        confidence = self._estimate_validation_confidence(max_batches=3)
        history = self.operator_score_memory.get(method, [])
        stability = float(np.std(history)) if len(history) >= 2 else 0.0

        reward = (1.0 - self.confidence_weight) * float(val_acc) + self.confidence_weight * confidence
        reward -= self.stability_lambda * stability

        if method in self.operator_ema_score:
            reward = self.reward_ema_decay * self.operator_ema_score[method] + (1.0 - self.reward_ema_decay) * reward

        self.operator_ema_score[method] = float(reward)
        self.operator_score_memory.setdefault(method, []).append(float(reward))
        return float(reward)

    def _fit_characterizer(self) -> DatasetCharacterizer:
        characterizer = DatasetCharacterizer()
        characteristics = []
        with torch.no_grad():
            for batch in self.train_loader:
                hsi = batch["hsi"].to(self.device)
                lidar = batch["lidar"].to(self.device)
                characteristics.append(characterizer.analyze(hsi, lidar))
        characterizer.fit_statistics(characteristics)
        self.control_system.characterizer.feature_stats = characterizer.feature_stats
        return characterizer

    def train_with_control(
        self,
        dataset_name: str,
        baseline_accuracy: float,
        sample_batch: Dict[str, torch.Tensor],
        max_iterations: int = 5,
    ) -> Tuple[float, List[Dict], List[Dict]]:
        self.model.set_feedback_refinement(False, gain=0.0)
        self.best_accuracy = float(baseline_accuracy)
        self.best_model_state = copy.deepcopy(self.model.state_dict())
        self.best_model_mode = self.model.active_mode
        self.best_model_method = "baseline_init"

        characterizer = self._fit_characterizer()
        hsi_sample = sample_batch["hsi"].to(self.device)
        lidar_sample = sample_batch["lidar"].to(self.device)

        characteristics = characterizer.analyze(hsi_sample, lidar_sample)
        target = AdaptiveTargetController().adjust_target(dataset_name, baseline_accuracy, characteristics)
        self.control_system.set_target(target)

        current_accuracy = float(baseline_accuracy)
        previous_accuracy = float(baseline_accuracy)
        self.acc_ema = float(baseline_accuracy)

        fusion_log: List[Dict] = []
        accuracy_log: List[Dict] = []
        bad_counter: Dict[FusionMode, int] = {}
        suppressed_until: Dict[FusionMode, int] = {}

        fallback_pool = [
            FusionMode.CROSS_MODAL_TRANSFORMER,
            FusionMode.PHYSICS_INFORMED,
            FusionMode.BASELINE_GHOST,
            FusionMode.BASELINE,
            FusionMode.ADAPTIVE_GATES,
            FusionMode.GEOMETRY_AWARE,
        ]

        for iteration in range(max_iterations):
            start_time = time.time()
            self.acc_ema = self.ema_lambda * self.acc_ema + (1.0 - self.ema_lambda) * current_accuracy
            methods, _, signal = self.control_system.control(hsi_sample, lidar_sample, self.acc_ema, dataset_name)

            refine_gain = float(np.clip(signal.magnitude, 0.05, 0.20))
            filtered = [mode for mode in methods if iteration >= suppressed_until.get(mode, -1)]
            for mode in fallback_pool:
                if len(filtered) >= 2:
                    break
                if mode not in filtered and iteration >= suppressed_until.get(mode, -1):
                    filtered.append(mode)
            if not filtered:
                filtered = [FusionMode.BASELINE_GHOST]

            best_iteration_score = -np.inf
            best_iteration_acc = -np.inf
            best_iteration_method: Optional[FusionMode] = None
            best_iteration_state: Optional[Dict[str, torch.Tensor]] = None
            tested_results: List[Tuple[str, float, float]] = []

            control_epochs = max(self.min_control_epochs, int(self.epochs * self.control_epochs_ratio))
            for method in filtered:
                self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_model_state.items()})
                self.model.set_fusion_mode(self.best_model_mode or FusionMode.BASELINE_GHOST)
                self.model.set_feedback_refinement(True, gain=refine_gain)

                result = self.train_with_mode(method, num_epochs=control_epochs)
                acc = float(result["best_acc"])
                val_std = float(result["val_std"])
                reward = self._compute_operator_reward(method, acc)
                selection_score = reward - 0.25 * val_std

                tested_results.append((method.value, acc, val_std))
                self.control_system.method_memory[method].append(float(reward))
                self.method_memory.setdefault(method, []).append(float(acc))

                if acc < current_accuracy - 0.02:
                    bad_counter[method] = bad_counter.get(method, 0) + 1
                    if bad_counter[method] >= 2:
                        suppressed_until[method] = iteration + 2
                else:
                    bad_counter[method] = 0

                if selection_score > best_iteration_score:
                    best_iteration_score = float(selection_score)
                    best_iteration_acc = float(acc)
                    best_iteration_method = method
                    best_iteration_state = copy.deepcopy(self.model.state_dict())

            accepted = False
            if (
                best_iteration_state is not None
                and best_iteration_method is not None
                and best_iteration_acc > current_accuracy + self.accept_margin
            ):
                accepted = True
                current_accuracy = best_iteration_acc
                self.best_accuracy = best_iteration_acc
                self.best_model_state = best_iteration_state
                self.best_model_mode = best_iteration_method
                self.best_model_method = best_iteration_method.value
                self.best_refinement_enabled = True
                self.best_refinement_gain = refine_gain

            self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_model_state.items()})
            self.model.set_fusion_mode(self.best_model_mode or FusionMode.BASELINE_GHOST)
            self.model.set_feedback_refinement(self.best_refinement_enabled, gain=self.best_refinement_gain)

            fusion_log.append(
                {
                    "iteration": iteration + 1,
                    "control_action": signal.action,
                    "selected_methods_raw": [mode.value for mode in methods],
                    "selected_methods_filtered": [mode.value for mode in filtered],
                    "tested_results": tested_results,
                    "best_method": best_iteration_method.value if best_iteration_method else None,
                    "best_iteration_acc": float(best_iteration_acc),
                    "best_iteration_score": float(best_iteration_score),
                    "accepted": bool(accepted),
                    "current_accuracy": float(current_accuracy),
                    "controller_error": float(signal.error),
                    "controller_magnitude": float(signal.magnitude),
                    "iteration_time_sec": float(time.time() - start_time),
                }
            )
            accuracy_log.append(
                {
                    "iteration": iteration + 1,
                    "accuracy": float(current_accuracy),
                    "best_method": self.best_model_method,
                }
            )

            if current_accuracy >= target:
                break
            if iteration > 1 and accepted and abs(current_accuracy - previous_accuracy) < 0.003:
                break
            previous_accuracy = current_accuracy

        return float(self.best_accuracy), fusion_log, accuracy_log


def create_controlled_model(
    hsi_channels: int,
    lidar_channels: int,
    num_classes: int,
    embed_dim: int = 256,
) -> ControlledFusionModel:
    return ControlledFusionModel(hsi_channels, lidar_channels, num_classes, embed_dim=embed_dim)
