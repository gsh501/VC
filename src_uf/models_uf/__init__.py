from .video_model import (
    DCVCUF,
    StreamlinedEntropyModel,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL_SCALE,
    UFModelConfig,
    UF_MODEL_CONFIGS,
    get_uf_model_config,
)
from .image_model import DCVCUFIntra

DMC = DCVCUF
DMCI = DCVCUFIntra

__all__ = [
    "DCVCUF",
    "DCVCUFIntra",
    "DMC",
    "DMCI",
    "StreamlinedEntropyModel",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_MODEL_SCALE",
    "UFModelConfig",
    "UF_MODEL_CONFIGS",
    "get_uf_model_config",
]
