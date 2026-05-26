"""Runtime configuration dataclasses for the legacy ROS-ABM entry path."""

from dataclasses import dataclass, field
from typing import Any


PERTURBATION_CELL_INDEX = 877
PERTURBATION_INJECTION_TIME = 5
PERTURBATION_INJECTION_H2O2I = 0.0012


@dataclass
class PerturbationConfig:
    enabled: bool = True
    cell_index: int = PERTURBATION_CELL_INDEX
    injection_time: int = PERTURBATION_INJECTION_TIME
    injection_h2o2i: float = PERTURBATION_INJECTION_H2O2I


@dataclass
class ModelConfig:
    ts: float = 0
    te: float = 300
    dt: float = 1
    x: float = 750
    y: float = 750
    z: float = 0
    ddim: float = 1.0
    sync_time: list = field(default_factory=lambda: [500, 500])
    num_cell_types: int = 2
    channel_count: float = 1000.0
    channel_permeability: float = 5e-9
    pluripotent_params: tuple = (4.89e-6, 0.03, 1.3e-7)
    differentiated_params: tuple = (6.0e-6, 0.04, 1.3e-7)
    open_probabilities: tuple = (0.45, 0.85)
    division_times_hours: tuple = (18, 51)
    D_camp: float = 444
    D_h2o2i: float = 1000
    D_h2o2e: float = 1800
    D_Ca: float = 760
    ddim_e: float = 1.0
    initial_h2o2e: float = 1e-9
    perturbation_cell_index: int = PERTURBATION_CELL_INDEX
    circle_radius: float = 50.0
    cell_radius: float = 6.5
    point_z: float = 0
    division_set: float = 0
    default_cell_state: str = "U"
    special_h2o2i: float = 1e-9
    special_camp: float = 1
    special_ca: float = 1e-7
    default_h2o2i: float = 1e-9
    default_camp_base: float = 0.595
    default_camp_random_scale: float = 0.05
    default_ca: float = 1e-7


@dataclass
class RunConfig:
    length: int = 1
    output_base_path: str = "~/Documents/PhD/test"
    inh_val: list = field(default_factory=lambda: [1.0])
    per_inh: list = field(default_factory=lambda: [1.0])
    inh_time: int = 1000
    inh_time1: int = 1000
    inhib_type: str | None = None
    magnitudes: list = field(default_factory=lambda: [1])
    chem_mode: str = "full"
    use_vectorized_hill: bool = False
    seed: int | None = None
    run_postprocess: bool = True
    model: ModelConfig = field(default_factory=ModelConfig)
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    run_name: Any = None
    output_path: str | None = None
    inhib_value: float | None = None
    magnitude: Any = None
    cell_path: str | None = None
