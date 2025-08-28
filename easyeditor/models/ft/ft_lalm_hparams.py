from dataclasses import dataclass
from typing import List, Optional
import yaml

from ...util.hparams import HyperParams


@dataclass
class FTLALMHyperParams(HyperParams):
    # Model
    name: str
    model_name: str
    model_class: str
    tokenizer_class: str
    tokenizer_name: str
    
    # Method
    num_steps: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float

    inner_params: List[str]
    device: int
    alg_name: str
    model_name: str

    # Defaults
    model_parallel: bool = False
    
    # wandb
    wandb_project: str = None
    wandb_run_name: Optional[str] = None
    wandb_enabled: bool = False
    
    # audio
    audio_root: str = ""
    cache_dir: str = './cache'

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'FT') or print(f'FTHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
