"""PyTorch implementation of the Prediction Interval Neural Network (PI-NN).

The network outputs three scalar values per sample:
    - ``ubs``: Upper bound of the prediction interval.
    - ``lbs``: Lower bound of the prediction interval.
    - ``regr``: Pointwise regression estimate.

Training optimizes a Quality-Driven (QD) style loss that balances:
    1. Interval coverage probability (``PICP_S`` >= ``confidence``).
    2. Minimum interval width on captured samples (``MPIW_c``).
    3. Pointwise regression accuracy (``w_MSE``) after an optional warm-up.
    4. Strict ordering of bounds (``lbs <= ubs``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.model.lr_schedule import LrSchedule

# Mapping from common string activation names to PyTorch module constructors.
ACTIVATION_REGISTRY: dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
}


def _resolve_activation(activation: str | Callable[[], nn.Module] | None) -> nn.Module | None:
    """Resolve an activation specification into an ``nn.Module`` instance."""
    if activation is None or activation == "linear":
        return None
    if isinstance(activation, str):
        key = activation.lower()
        if key not in ACTIVATION_REGISTRY:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Supported: {list(ACTIVATION_REGISTRY.keys())}"
            )
        return ACTIVATION_REGISTRY[key]()
    return activation()


class PiNeuralNetwork(nn.Module):
    """Neural network for simultaneous Prediction Interval and regression estimation.

    Args:
        input_dim: Number of input features.
        hidden_layers: Sequence of ``(units, activation)`` pairs defining the MLP backbone.
        confidence: Target coverage probability in ``(0, 1)`` (e.g. ``0.95``).
        ub_bias_init: Initial bias for the upper-bound output neuron.
        lb_bias_init: Initial bias for the lower-bound output neuron.
        softening_factor: Steepness of the sigmoid approximation used in ``PICP_S``.
        lr: Initial learning rate for the AdamW optimizer.
        weight_decay: Decoupled L2 weight decay coefficient for AdamW.
        alpha: Weight applied to the MSE regression term after warm-up.
        lambda_: Weight applied to the under-coverage penalty.
        mse_inactivity_epochs: Number of initial epochs during which ``w_MSE`` is muted
            (warm-up phase focused solely on interval bounds).
        lr_schedule: Optional learning-rate schedule callable.
        device: Target device (defaults to CUDA if available, otherwise CPU).
        verbose: Whether to print per-epoch training loss.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: Iterable[tuple[int, str | Callable[[], nn.Module] | None]],
        confidence: float = 0.95,
        ub_bias_init: float = 1.0,
        lb_bias_init: float = -1.0,
        softening_factor: float = 120.0,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        alpha: float = 10.0,
        lambda_: float = 0.005,
        mse_inactivity_epochs: int = 0,
        uncertainty_weighted_mse: bool = True,
        lr_schedule: LrSchedule | None = None,
        device: str | torch.device | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__()

        if not (0.0 < confidence < 1.0):
            raise ValueError(f"confidence must be in (0, 1), got {confidence}.")

        # Build the sequential MLP backbone dynamically from hidden_layers.
        layers: list[nn.Module] = []
        in_features = input_dim

        for units, activation_spec in hidden_layers:
            linear = nn.Linear(in_features, units)
            # Match original initialization: N(0, 0.2) weights, zero biases.
            nn.init.normal_(linear.weight, mean=0.0, std=0.2)
            nn.init.zeros_(linear.bias)
            layers.append(linear)

            act_module = _resolve_activation(activation_spec)
            if act_module is not None:
                layers.append(act_module)

            in_features = units

        # Final linear head with 3 outputs: [upper_bound, lower_bound, regression].
        output_layer = nn.Linear(in_features, 3)
        nn.init.normal_(output_layer.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            output_layer.bias.copy_(torch.tensor([ub_bias_init, lb_bias_init, 1.0]))
        layers.append(output_layer)

        self.backbone = nn.Sequential(*layers)

        # Hyperparameters for the QD-style loss and training loop.
        self.confidence = confidence
        self.softening_factor = softening_factor
        self.alpha = alpha
        self.lambda_ = lambda_
        self.mse_inactivity_epochs = mse_inactivity_epochs
        self.uncertainty_weighted_mse = uncertainty_weighted_mse
        self.epsilon = 1e-3
        self.lr_schedule = lr_schedule
        self.verbose = verbose
        self.epoch = 0

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.to(self.device)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute upper bound, lower bound, and regression estimate for input ``x``.

        Returns:
            Tuple ``(ubs, lbs, regr)``, each of shape ``(batch_size,)``.
        """
        outputs = self.backbone(x)
        ubs, lbs, regr = outputs.unbind(dim=-1)
        return ubs, lbs, regr

    def compute_loss(
        self,
        targets: torch.Tensor,
        ubs: torch.Tensor,
        lbs: torch.Tensor,
        regr: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Quality-Driven Prediction Interval loss.

        Step-by-step breakdown:
            1. ``K_S`` (soft coverage mask): smooth sigmoid approximation of
               ``lbs <= targets <= ubs`` so gradients flow through coverage.
            2. ``K_H`` (hard coverage mask): exact indicator for samples inside
               ``(lbs, ubs)`` used to isolate captured intervals.
            3. ``MPIW_c``: mean interval width computed only on captured samples.
            4. ``PICP_S``: soft batch coverage probability.
            5. ``w_MSE``: weighted regression MSE, muted during warm-up epochs.
            6. ``Loss_S``: combines ``w_MSE``, ``MPIW_c``, an under-coverage
               hinge penalty scaled by binomial variance, and a heavy penalty
               against bound inversion (``lbs > ubs``).
        """
        targets = targets.reshape(-1)
        batch_size = targets.shape[0]

        # 1. Differentiable (soft) coverage mask via scaled sigmoids.
        k_su = torch.sigmoid(self.softening_factor * (ubs - targets))
        k_sl = torch.sigmoid(self.softening_factor * (targets - lbs))
        k_s = k_su * k_sl

        # 2. Non-differentiable (hard) coverage mask for captured points.
        k_hu = torch.clamp(torch.sign(ubs - targets), min=0.0)
        k_hl = torch.clamp(torch.sign(targets - lbs), min=0.0)
        k_h = k_hu * k_hl

        # 3. Mean Prediction Interval Width on captured samples (epsilon avoids /0).
        mpiw_c = torch.sum(torch.abs(ubs - lbs) * (k_h + self.epsilon / batch_size)) / (
            torch.sum(k_h) + self.epsilon
        )

        # 4. Soft Prediction Interval Coverage Probability.
        picp_s = torch.mean(k_s)

        # 5. Pointwise regression MSE, active only after warm-up.
        # Optionally divide per-sample squared error by detached interval width
        # so high-variance regions (wide intervals) cannot pull the regression fit.
        mse_weight = 0.0 if self.epoch < self.mse_inactivity_epochs else self.alpha
        sq_error = (regr - targets) ** 2
        if self.uncertainty_weighted_mse:
            interval_width = torch.clamp((ubs - lbs).detach(), min=self.epsilon)
            sq_error = sq_error / interval_width
        w_mse = mse_weight * torch.mean(sq_error)

        # 6. Total loss: regression + width + under-coverage penalty + bound-order penalty.
        coverage_penalty = (
            self.lambda_
            * batch_size
            / ((1.0 - self.confidence) * self.confidence)
            * torch.clamp(self.confidence - picp_s, min=0.0)
        )
        # Quadratic penalty so the gradient is 0 at lbs == ubs (avoids violent rebounds)
        # while still strongly correcting inverted bounds (lbs > ubs).
        inversion_penalty = 100.0 * torch.mean(F.relu(lbs - ubs) ** 2)

        return w_mse + mpiw_c + coverage_penalty + inversion_penalty

    def fit(
        self,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        epochs: int = 100,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> list[float]:
        """Train the model on ``(x, y)`` and return the per-epoch mean loss history."""
        x_tensor = torch.as_tensor(x, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.float32).reshape(-1)

        if x_tensor.ndim == 1:
            x_tensor = x_tensor.unsqueeze(-1)

        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=len(dataset) >= batch_size,
        )

        history: list[float] = []
        self.train()

        for _ in range(1, epochs + 1):
            epoch_losses: list[float] = []

            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                ubs, lbs, regr = self(batch_x)
                loss = self.compute_loss(targets=batch_y, ubs=ubs, lbs=lbs, regr=regr)
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())

            mean_loss = float(np.mean(epoch_losses))
            history.append(mean_loss)
            self.epoch += 1

            if self.lr_schedule is not None:
                current_lr = self.optimizer.param_groups[0]["lr"]
                new_lr = self.lr_schedule(self.epoch, current_lr)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = new_lr

            if self.verbose and (self.epoch % 50 == 0 or self.epoch == 1):
                print(f"Epoch {self.epoch:4d} | Loss: {mean_loss:.4f}")

        return history

    @torch.no_grad()
    def predict(self, x: np.ndarray | torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference and return ``(ubs, lbs, regr)`` as NumPy arrays."""
        self.eval()
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if x_tensor.ndim == 1:
            x_tensor = x_tensor.unsqueeze(-1)

        ubs, lbs, regr = self(x_tensor)
        return ubs.cpu().numpy(), lbs.cpu().numpy(), regr.cpu().numpy()
