import torch
import copy
import json
import logging
import transformers
from typing import Dict, Optional, Sequence
from datasets import load_dataset
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig, AutoModelForCausalLM
from LWM.modeling_llama_wm import WMLlamaForCausalLM, AllInOneModel
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

IGNORE_INDEX = -100

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s for s in sources]
    examples_tokenized = _tokenize_fn(examples, tokenizer)
    input_ids = examples_tokenized["input_ids"]
    return dict(input_ids=input_ids)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = None
        if data_path.strip().split(".")[-1] == "jsonl":
            with open(data_path) as f:
                list_data_dict = [json.loads(line) for line in f]
        else:
            list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        sources = [
            example['instruction'] for example in list_data_dict
        ]
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.input_ids[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    # def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    def __call__(self, instances):
        input_ids = tuple(instance["input_ids"].flip(dims=[0]) for instance in instances)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).flip(dims=[1])
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        return dict(
            input_ids=input_ids,
            labels=input_ids,
            attention_mask=attention_mask
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    eval_dataset = None
    if data_args.eval_data_path is not None:
        eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.eval_data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def get_model_tokenizer(training_args, model_args, other_args):
    model = None
    bnb_config = None
    if other_args.load_in_8bit == True or other_args.load_in_4bit == True \
        or training_args.bf16 == True or training_args.fp16 == True:
        raise ValueError(
            f"There is an unknown bug: load model in 8bit or 4bit will cause none grad"
            " please disable all quantization methods"
        )

    if other_args.load_in_8bit == True or other_args.load_in_4bit == True:
        load_in_4bit = other_args.load_in_4bit
        load_in_8bit = False if load_in_4bit else other_args.load_in_8bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    # "/home/hl3352/LLMs/LearnableWatermarking/learnable_watermarking/exp_wikitext/save_model_alpha0.95_ne5_llama2_7b_qvko_r8_a16_lr1e-4_bs0/checkpoint-8",
    print(f"model: {model_args.model_name_or_path}")
    model = WMLlamaForCausalLM.from_pretrained(
    # model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
    )

    # BUG: use prepare_model_for_kbit_training will change the model behavior
#     if other_args.load_in_8bit == True or other_args.load_in_4bit == True:
#         model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.eos_token

    if other_args.use_lora == True:
        config = LoraConfig(
            r=other_args.lora_r,
            lora_alpha=other_args.lora_alpha,
            target_modules = other_args.lora_target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights=False,
        )
        print("Use LoRA:", config)

        model.enable_input_require_grads()
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        model.config.use_cache = False
        model.is_parallelizable = True
        model.model_parallel = True

    model = AllInOneModel(model, tokenizer, other_args.loss_alpha)
#     for param_tensor in model.state_dict():
#         print(param_tensor, "\t", model.state_dict()[param_tensor].size(), model.state_dict()[param_tensor].sum())
#     exit()


    # TODO: load a checkpoint
#     if training_args.resume_from_checkpoint:
#         # Check the available weights and load them
#         checkpoint_name = os.path.join(
#             training_args.resume_from_checkpoint, "pytorch_model.bin"
#         )  # Full checkpoint
#         if not os.path.exists(checkpoint_name) and other_args.use_lora:
#             checkpoint_name = os.path.join(
#                 training_args.resume_from_checkpoint, "adapter_model.bin"
#             )  # only LoRA model - LoRA config above has to fit
#             training_args.resume_from_checkpoint = (
#                 False  # So the trainer won't try loading its state
#             )
#         # The two files above have a different name depending on how they were saved, but are actually the same.
#         if os.path.exists(checkpoint_name):
#             print(f"Restarting from {checkpoint_name}")
#             adapters_weights = torch.load(checkpoint_name)
#             set_peft_model_state_dict(model, adapters_weights)
#         else:
#             print(f"Checkpoint {checkpoint_name} not found")
    return model, tokenizer
