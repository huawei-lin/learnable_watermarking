import os
import copy
import json
import logging
from typing import Dict, Optional, Sequence
from datasets import load_dataset
from LWM.loaders import make_supervised_data_module, get_model_tokenizer
from LWM.utils import get_args

import torch
import transformers
from LWM.trainer import WatermarkTrainer
torch.manual_seed(42)

def train():
    model_args, data_args, training_args, other_args = get_args()
    print(other_args, other_args.use_lora)

    model, tokenizer = get_model_tokenizer(training_args, model_args, other_args)
    model = model.to(0)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = WatermarkTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

#     if other_args.use_lora and torch.__version__ >= "2":
#         model = torch.compile(model)

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
