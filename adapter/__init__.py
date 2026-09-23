"""ShiftProof Model Adapters."""

from adapter.base import ModelAdapter
from adapter.mobilenet_v2_adapter import MobileNetV2Adapter
from adapter.squeezenet_adapter import SqueezeNetAdapter

__all__ = [
    "ModelAdapter",
    "MobileNetV2Adapter",
    "SqueezeNetAdapter",
]
