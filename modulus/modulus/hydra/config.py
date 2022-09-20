"""
Modulus main config
"""

from platform import architecture
import torch
import logging
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from hydra.conf import RunDir, HydraConf
from omegaconf import MISSING, SI
from typing import List, Any
from modulus.constants import JIT_PYTORCH_VERSION

from .loss import LossConf
from .optimizer import OptimizerConf
from .pde import PDEConf
from .scheduler import SchedulerConf
from .training import TrainingConf
from .profiler import ProfilerConf
from .hydra import default_hydra

logger = logging.getLogger(__name__)


@dataclass
class ModulusConfig:

    # General parameters
    network_dir: str = "."
    initialization_network_dir: str = ""
    save_filetypes: str = "vtk"
    summary_histograms: bool = False
    jit: bool = bool(torch.__version__ == JIT_PYTORCH_VERSION)
    jit_use_nvfuser: bool = False

    cuda_graphs: bool = True
    cuda_graph_warmup: int = 20
    find_unused_parameters: bool = False
    broadcast_buffers: bool = False

    device: str = ""
    debug: bool = False
    run_mode: str = "train"

    arch: Any = MISSING  # List of archs

    training: TrainingConf = MISSING
    loss: LossConf = MISSING

    optimizer: OptimizerConf = MISSING
    scheduler: SchedulerConf = MISSING

    batch_size: Any = MISSING
    profiler: ProfilerConf = MISSING

    hydra: Any = field(default_factory=lambda: default_hydra)

    # User custom parameters that are not used internally in modulus
    custom: Any = MISSING


default_defaults = [
    {"training": "default_training"},
    {"profiler": "nvtx"},
    {"override hydra/job_logging": "info_logging"},
    {"override hydra/launcher": "basic"},
    {"override hydra/help": "modulus_help"},
    {"override hydra/callbacks": "default_callback"},
]


@dataclass
class DefaultModulusConfig(ModulusConfig):
    # Core defaults
    # Can over-ride default with "override" hydra command
    defaults: List[Any] = field(default_factory=lambda: default_defaults)


# Modulus config for debugging
debug_defaults = [
    {"training": "default_training"},
    {"profiler": "nvtx"},
    {"override hydra/job_logging": "debug_logging"},
    {"override hydra/help": "modulus_help"},
    {"override hydra/callbacks": "default_callback"},
]


@dataclass
class DebugModulusConfig(ModulusConfig):
    # Core defaults
    # Can over-ride default with "override" hydra command
    defaults: List[Any] = field(default_factory=lambda: debug_defaults)

    debug: bool = True


# Modulus config with experimental features (use caution)
experimental_defaults = [
    {"training": "default_training"},
    {"profiler": "nvtx"},
    {"override hydra/job_logging": "info_logging"},
    {"override hydra/launcher": "basic"},
    {"override hydra/help": "modulus_help"},
    {"override hydra/callbacks": "default_callback"},
]


@dataclass
class ExperimentalModulusConfig(ModulusConfig):
    # Core defaults
    # Can over-ride default with "override" hydra command
    defaults: List[Any] = field(default_factory=lambda: experimental_defaults)

    pde: PDEConf = MISSING


def register_modulus_configs() -> None:

    if not torch.__version__ == JIT_PYTORCH_VERSION:
        logger.warn(
            f"TorchScript default is being turned off due to PyTorch version mismatch."
        )

    cs = ConfigStore.instance()
    cs.store(
        name="modulus_default",
        node=DefaultModulusConfig,
    )
    cs.store(
        name="modulus_debug",
        node=DebugModulusConfig,
    )
    cs.store(
        name="modulus_experimental",
        node=ExperimentalModulusConfig,
    )
