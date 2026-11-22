"""Simple learning-rate schedules compatible with the PyTorch training loop."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


class LrSchedule(ABC):
    """Abstract base class for epoch-based learning-rate schedules."""

    @abstractmethod
    def __call__(self, epoch: int, current_lr: float) -> float:
        """Return the updated learning rate for the given ``epoch``."""


class TimeDecreasingLr(LrSchedule):
    """Inverse-time learning-rate decay: ``initial_lr / (1 + decay * epoch)``."""

    def __init__(self, decay: float, initial_lr: float | None = None) -> None:
        self.decay = decay
        self._initial_lr = initial_lr

    def __call__(self, epoch: int, current_lr: float) -> float:
        # Capture the starting learning rate on the first call so decay is
        # computed relative to initial_lr rather than compounding every epoch.
        if self._initial_lr is None:
            self._initial_lr = current_lr
        return float(self._initial_lr / (1.0 + self.decay * epoch))


class StepDecreasingLr(LrSchedule):
    """Stepwise exponential decay every ``decay_after`` epochs."""

    def __init__(self, initial_lr: float, decay: float, decay_after: int) -> None:
        self.init_lr = initial_lr
        self.decay = decay
        self.decay_after = decay_after

    def __call__(self, epoch: int, current_lr: float = 0.0) -> float:
        return float(
            self.init_lr * math.pow(self.decay, math.floor((1 + epoch) / self.decay_after))
        )
