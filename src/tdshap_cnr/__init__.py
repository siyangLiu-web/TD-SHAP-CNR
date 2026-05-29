"""
TD-SHAP Causal Network Reconstruction (tdshap_cnr)
A toolkit for dynamic causal network modeling and topology evaluation.
"""

from .config import (
    PipelineConfig,
    ModelConfig,
    PruneConfig,
    DynamicConfig,
    EvalConfig
)
from .pipeline import CausalNetworkPipeline

__version__ = "0.1.0"

__all__ = [
    "PipelineConfig",
    "ModelConfig",
    "PruneConfig",
    "DynamicConfig",
    "EvalConfig",
    "CausalNetworkPipeline"
]