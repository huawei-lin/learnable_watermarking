import argparse
import distutils.util
from typing import Dict, Optional, Sequence
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


def get_args():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args, other_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    other_parser = argparse.ArgumentParser()
    other_parser.add_argument('--load_in_4bit', type=lambda x:bool(distutils.util.strtobool(x)), default=False)
    other_parser.add_argument('--load_in_8bit', type=lambda x:bool(distutils.util.strtobool(x)), default=False)
    other_parser.add_argument('--use_lora', type=lambda x:bool(distutils.util.strtobool(x)), default=False)
    other_parser.add_argument('--lora_r', type=int, default=8)
    other_parser.add_argument('--lora_alpha', type=int, default=16)
    other_parser.add_argument('--lora_target_modules', nargs="+", default=['q_proj, v_proj'])
    other_args = other_parser.parse_args(other_args)
    return model_args, data_args, training_args, other_args
