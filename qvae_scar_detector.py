"""Quantum variational autoencoder for seed-state family detection.

This module contains only the QVAE model, training, and inference logic.  It
does not construct or diagonalise a Hamiltonian.  Callers supply statevectors,
for example eigenvectors produced by the repository's existing exact-
diagonalisation tools.

PennyLane and PyTorch are optional repository dependencies.  Importing this
module does not require them, but constructing :class:`QVAEScarDetector` does.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:  # Keep the rest of the repository importable without QML dependencies.
    import pennylane as qml
except ImportError:  # pragma: no cover - depends on the user's environment.
    qml = None  # type: ignore[assignment]

try:
    import torch
except ImportError:  # pragma: no cover - depends on the user's environment.
    torch = None  # type: ignore[assignment]


_COST_TOLERANCE = 1.0e-10


@dataclass(frozen=True)
class QVAEConfig:
    """Configuration for a statevector QVAE.

    The final ``n_trash`` wires are trash wires; all preceding wires are
    latent wires.  ``convergence_tolerance`` is the minimum cost improvement
    that resets the early-stopping counter, and ``patience`` is the number of
    consecutive optimizer steps without such an improvement.
    """

    n_qubits: int
    n_trash: int
    n_layers: int
    learning_rate: float
    steps: int
    seed: int
    device_name: str = "default.qubit"
    convergence_tolerance: float = 1.0e-8
    patience: int = 50

    def __post_init__(self) -> None:
        """Validate all configuration fields."""
        for name in ("n_qubits", "n_trash", "n_layers", "steps", "seed", "patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer.")
            object.__setattr__(self, name, int(value))

        if self.n_qubits < 2:
            raise ValueError("n_qubits must be at least 2.")
        if not 1 <= self.n_trash < self.n_qubits:
            raise ValueError("n_trash must satisfy 1 <= n_trash < n_qubits.")
        if self.n_layers < 1:
            raise ValueError("n_layers must be at least 1.")
        if self.steps < 1:
            raise ValueError("steps must be at least 1.")
        if self.patience < 1:
            raise ValueError("patience must be at least 1.")

        for name in ("learning_rate", "convergence_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number.")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.convergence_tolerance < 0.0:
            raise ValueError("convergence_tolerance must be non-negative.")
        if not isinstance(self.device_name, str):
            raise TypeError("device_name must be a string.")
        if not self.device_name.strip():
            raise ValueError("device_name must be non-empty.")

    @property
    def n_latent(self) -> int:
        """Number of latent wires retained by the encoder."""
        return self.n_qubits - self.n_trash

    @property
    def latent_wires(self) -> tuple[int, ...]:
        """Latent-wire indices, ordered from the first physical wire."""
        return tuple(range(self.n_latent))

    @property
    def trash_wires(self) -> tuple[int, ...]:
        """Trash-wire indices, comprising the final configured wires."""
        return tuple(range(self.n_latent, self.n_qubits))


@dataclass(frozen=True)
class QVAETrainingResult:
    """Summary and reproducible outputs of one QVAE training run.

    ``final_cost`` is the cost after the final optimizer step, whereas
    ``best_cost`` and ``optimized_parameters`` describe the best point seen
    during training.  The detector itself retains those best parameters.
    """

    initial_cost: float
    final_cost: float
    best_cost: float
    best_step: int
    steps_completed: int
    converged: bool
    cost_history: tuple[float, ...]
    optimized_parameters: np.ndarray
    config: QVAEConfig


class QVAEScarDetector:
    """Train and apply a QVAE encoder to a family of statevectors.

    The encoder is trained on one seed state.  Its normalized expected Hamming
    weight on the trash wires is the QVAE cost.  ``similarity = 1 - cost`` is
    therefore similarity to the training seed under the learned compression;
    it is **not** a calibrated probability that an input state is a scar.

    Parameters
    ----------
    config:
        Validated circuit and optimization configuration.
    """

    def __init__(self, config: QVAEConfig) -> None:
        if not isinstance(config, QVAEConfig):
            raise TypeError("config must be a QVAEConfig instance.")
        self._require_dependencies()
        self.config = config
        self._parameters: Any | None = None

        try:
            self._device = qml.device(config.device_name, wires=config.n_qubits)
        except Exception as error:
            raise ValueError(
                f"Could not create PennyLane device {config.device_name!r} "
                f"for {config.n_qubits} qubits."
            ) from error

        @qml.qnode(self._device, interface="torch", diff_method="backprop")
        def qvae_circuit(state: Any, parameters: Any) -> tuple[Any, ...]:
            qml.StatePrep(state, wires=range(config.n_qubits), normalize=False)
            for layer in range(config.n_layers):
                for wire in range(config.n_qubits):
                    qml.RY(parameters[layer, wire], wires=wire)
                start = layer % 2
                for wire in range(start, config.n_qubits - 1, 2):
                    qml.CZ(wires=(wire, wire + 1))
            return tuple(qml.expval(qml.PauliZ(wire)) for wire in config.trash_wires)

        self._circuit = qvae_circuit

    @staticmethod
    def _require_dependencies() -> None:
        """Raise an actionable error if optional QVAE packages are absent."""
        missing = []
        if qml is None:
            missing.append("PennyLane")
        if torch is None:
            missing.append("PyTorch")
        if missing:
            names = " and ".join(missing)
            raise ImportError(
                f"QVAEScarDetector requires {names}. Install the optional "
                "QVAE dependencies before constructing a detector."
            )

    def _prepare_state(self, state: np.ndarray) -> Any:
        """Validate, normalize, and convert one state to a Torch tensor."""
        if not isinstance(state, np.ndarray):
            raise TypeError("state must be a NumPy array.")
        expected_shape = (1 << self.config.n_qubits,)
        if state.shape != expected_shape:
            raise ValueError(f"state must have shape {expected_shape}, got {state.shape}.")
        if not np.issubdtype(state.dtype, np.number):
            raise TypeError("state must contain real or complex numeric amplitudes.")

        amplitudes = np.asarray(state, dtype=np.complex128)
        if not np.all(np.isfinite(amplitudes)):
            raise ValueError("state contains non-finite amplitudes.")
        norm = float(np.linalg.norm(amplitudes))
        if not math.isfinite(norm) or norm == 0.0:
            raise ValueError("state must have a finite, non-zero norm.")
        return torch.as_tensor(amplitudes / norm, dtype=torch.complex128)

    def _prepare_parameters(self, parameters: Any | None) -> Any:
        """Resolve and validate parameters for an inference call."""
        if parameters is None:
            if self._parameters is None:
                raise RuntimeError("No trained parameters are available; call fit() first.")
            return self._parameters

        tensor = torch.as_tensor(parameters, dtype=torch.float64)
        expected_shape = (self.config.n_layers, self.config.n_qubits)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"parameters must have shape {expected_shape}, got {tuple(tensor.shape)}."
            )
        if not bool(torch.all(torch.isfinite(tensor)).item()):
            raise ValueError("parameters contain non-finite values.")
        return tensor

    def _cost_tensor(self, state: Any, parameters: Any) -> Any:
        """Evaluate the differentiable mean trash-wire excitation."""
        measurements = self._circuit(state, parameters)
        if isinstance(measurements, (tuple, list)):
            z_values = torch.stack(tuple(measurements))
        else:
            z_values = measurements.reshape(1)
        return torch.mean((1.0 - z_values) / 2.0)

    @staticmethod
    def _checked_cost(value: Any, context: str) -> float:
        """Convert a scalar cost to float and enforce its physical range."""
        numeric = float(value.detach().cpu().item())
        if not math.isfinite(numeric):
            raise FloatingPointError(f"Non-finite QVAE cost encountered {context}.")
        if numeric < -_COST_TOLERANCE or numeric > 1.0 + _COST_TOLERANCE:
            raise FloatingPointError(
                f"QVAE cost {numeric} lies outside [0, 1] {context}."
            )
        return min(1.0, max(0.0, numeric))

    def fit(self, seed_state: np.ndarray) -> QVAETrainingResult:
        """Train the encoder on one seed state with deterministic Adam.

        The best parameters encountered, including the initialization, are
        retained for subsequent calls.  Early stopping occurs after
        ``patience`` consecutive steps without an improvement greater than
        ``convergence_tolerance``.
        """
        state = self._prepare_state(seed_state)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed)
        parameters = (
            2.0
            * math.pi
            * (torch.rand(
                (self.config.n_layers, self.config.n_qubits),
                dtype=torch.float64,
                generator=generator,
            ) - 0.5)
        ).requires_grad_(True)
        optimizer = torch.optim.Adam([parameters], lr=self.config.learning_rate)

        with torch.no_grad():
            initial = self._checked_cost(
                self._cost_tensor(state, parameters), "at initialization"
            )
        history = [initial]
        best_cost = initial
        best_step = 0
        best_parameters = parameters.detach().clone()
        stopping_reference = initial
        stale_steps = 0
        converged = False
        steps_completed = 0

        for step in range(1, self.config.steps + 1):
            optimizer.zero_grad()
            loss = self._cost_tensor(state, parameters)
            self._checked_cost(loss, f"before optimizer step {step}")
            loss.backward()
            if parameters.grad is None or not bool(torch.all(torch.isfinite(parameters.grad)).item()):
                raise FloatingPointError(
                    f"Non-finite or missing QVAE gradient at optimizer step {step}."
                )
            optimizer.step()

            with torch.no_grad():
                current = self._checked_cost(
                    self._cost_tensor(state, parameters), f"after optimizer step {step}"
                )
            history.append(current)
            steps_completed = step

            if current < best_cost:
                best_cost = current
                best_step = step
                best_parameters = parameters.detach().clone()

            if stopping_reference - current > self.config.convergence_tolerance:
                stopping_reference = current
                stale_steps = 0
            else:
                stale_steps += 1
                if stale_steps >= self.config.patience:
                    converged = True
                    break

        self._parameters = best_parameters
        return QVAETrainingResult(
            initial_cost=initial,
            final_cost=history[-1],
            best_cost=best_cost,
            best_step=best_step,
            steps_completed=steps_completed,
            converged=converged,
            cost_history=tuple(history),
            optimized_parameters=best_parameters.cpu().numpy().copy(),
            config=self.config,
        )

    def cost(self, state: np.ndarray, parameters: Any | None = None) -> float:
        """Return normalized expected Hamming weight on the trash wires."""
        prepared_state = self._prepare_state(state)
        prepared_parameters = self._prepare_parameters(parameters)
        with torch.no_grad():
            value = self._cost_tensor(prepared_state, prepared_parameters)
        return self._checked_cost(value, "during inference")

    def similarity(self, state: np.ndarray, parameters: Any | None = None) -> float:
        """Return ``1 - cost``, a learned similarity to the training seed.

        This quantity is not a calibrated probability that the state is a
        quantum many-body scar.
        """
        return 1.0 - self.cost(state, parameters=parameters)

    def scan(self, states: Iterable[np.ndarray]) -> list[dict[str, float | int]]:
        """Score supplied states in iteration order using retained parameters."""
        if self._parameters is None:
            raise RuntimeError("No trained parameters are available; call fit() first.")
        rows: list[dict[str, float | int]] = []
        for index, state in enumerate(states):
            qvae_cost = self.cost(state)
            rows.append(
                {
                    "state_index": index,
                    "qvae_cost": qvae_cost,
                    "qvae_similarity": 1.0 - qvae_cost,
                }
            )
        return rows

    def save_checkpoint(self, path: str | Path) -> None:
        """Save retained best parameters and configuration to one NPZ file."""
        if self._parameters is None:
            raise RuntimeError("No trained parameters are available to save.")
        destination = Path(path)
        if destination.parent != Path(""):
            destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(self.config), sort_keys=True)
        with destination.open("wb") as stream:
            np.savez_compressed(
                stream,
                format_version=np.array(1, dtype=np.int64),
                config=np.array(payload),
                parameters=self._parameters.detach().cpu().numpy(),
            )

    def load_checkpoint(self, path: str | Path) -> None:
        """Load parameters from a compatible checkpoint into this detector.

        The checkpoint configuration must exactly match this detector's
        configuration, preventing accidental use with a different wire layout
        or circuit architecture.
        """
        source = Path(path)
        try:
            with np.load(source, allow_pickle=False) as checkpoint:
                version = int(checkpoint["format_version"])
                stored_config = QVAEConfig(**json.loads(str(checkpoint["config"])))
                parameters = np.asarray(checkpoint["parameters"], dtype=np.float64)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid QVAE checkpoint: {source}") from error
        if version != 1:
            raise ValueError(f"Unsupported QVAE checkpoint format version: {version}.")
        if stored_config != self.config:
            raise ValueError(
                "Checkpoint configuration does not match this detector's configuration."
            )
        self._parameters = self._prepare_parameters(parameters).detach().clone()


__all__ = ["QVAEConfig", "QVAETrainingResult", "QVAEScarDetector"]
