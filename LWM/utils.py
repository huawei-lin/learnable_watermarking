import argparse
import distutils.util
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
import transformers

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default=None, metadata={"help": "Path to the eval data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


@dataclass
class AdapterArguments:
    load_in_4bit: Optional[bool] = field(default=False)
    load_in_8bit: Optional[bool] = field(default=False)
    use_lora: Optional[bool] = field(default=True)
    lora_r: Optional[int] = field(default=8)
    lora_alpha: Optional[int] = field(default=16)
    lora_target_modules: Optional[List[str]] = field(default_factory=lambda: ['q_proj', 'v_proj'])
    loss_alpha: Optional[float] = field(default=0.5)


def get_args():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, AdapterArguments))
    model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()
    return model_args, data_args, training_args, adapter_args
