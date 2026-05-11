"""Model components for variational inference in stable-driven SDEs.

This package provides modular components for building stochastic differential
equations (SDEs) with jump processes, including:

- Drift components: Deterministic evolution terms
- Scale/diffusion components: Noise scaling factors
- Attention mechanisms: Temporal encodings that learn important time points
- Potential functions: Quadratic potentials for exponential tilting of jump measures
- Utility functions: Fourier features, parameter scaling, etc.
"""

# Drift components
from models.components.drift import (
    DiagonalLinearFunction,
    OUDiagonalLinearFunction,
    NeuralNetFunction,
    DoubleWellDriftFunction,
)

# Scale/diffusion components
from models.components.scale import (
    DiagonalScaleFactor,
    CholeskyDiffusion,
)

# Attention mechanisms
from models.components.attention import (
    TemporalAttentionEncoding,
)

# Potential functions
from models.components.potential import (
    QuadraticNeuralPotential,
)

# Utility functions
from models.components.utils import (
    fourier_features,
    scale_mlp_params,
)

__all__ = [
    # Drift
    "DiagonalLinearFunction",
    "OUDiagonalLinearFunction",
    "NeuralNetFunction",
    "DoubleWellDriftFunction",
    # Scale
    "DiagonalScaleFactor",
    "CholeskyDiffusion",
    # Attention
    "TemporalAttentionEncoding",
    # Potential
    "QuadraticNeuralPotential",
    # Utils
    "fourier_features",
    "scale_mlp_params",
]
