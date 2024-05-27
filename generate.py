import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import copy
import logging
import numpy as np
np.set_printoptions(precision=2)
import transformers
import torch
import torch.nn.functional as F
from LWM.loaders import get_model_tokenizer
from LWM.utils import (
    ModelArguments,
    TrainingArguments,
    AdapterArguments
)

# prompt = "Question: Where is Singapore? Answer:"
# prompt = "America is the"
# prompt = "China is"
# prompt = "China is a major economic powerhouse for the world economy."
# prompt = "In modern neuroscience, observing the dynamics of large populations of neurons is a critical step of understanding how networks of neurons process information. Light-field microscopy (LFM) has emerged as a type of scanless, high-speed, three-dimensional (3D) imaging tool, particularly attractive for this purpose."
# prompt = "In modern neuroscience, observing the dynamics of large populations of neurons is a critical step of understanding how networks of neurons process information."
# prompt = "Recently, Large language models (LLMs) have"


base_model_path = "meta-llama/Llama-2-7b-hf"
base_model_path = "meta-llama/Meta-Llama-3-8B"

checkpoints_path = "/home/hl3352/LLMs/LearnableWatermarking/learnable_watermarking/exp_wikitext/stage_limit_decay0.99_wiki_t16_3L_dense_alpha0.5_ne1_llama3_8b_qv_r8_a16_lr1e-4_bs0/checkpoint-1440"


lora_r  = 8
lora_alpha = lora_r*2

model_args = ModelArguments
model_args.model_name_or_path = base_model_path

training_args= TrainingArguments
training_args.discriminator_resume_from_checkpoint = checkpoints_path
training_args.adapters_resume_from_checkpoint = checkpoints_path

adapter_args = AdapterArguments
adapter_args.lora_r = lora_r
adapter_args.lora_alpha = lora_alpha
adapter_args.lora_target_modules = ['q_proj', 'v_proj']

model, tokenizer = get_model_tokenizer(training_args, model_args, adapter_args)
model = model.cuda()
print(model)

model.eval()
tokenized = tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            max_length=512,
            truncation=True,
        )
input_ids = tokenized.input_ids.cuda()
# input_ids = input_ids[0].repeat(4, 1)
print("input_ids:", input_ids)


with model.disable_adapter():
    base_generation = model.generate(
        input_ids,
        max_length=512,
        # attention_mask=attention_mask,
        num_beams=5,
        no_repeat_ngram_size=4,
        num_return_sequences=1,
        do_sample = True,
        top_p = 0.8,
        pad_token_id=tokenizer.pad_token_id,
        early_stopping=True,
    )
    base_model_generation = tokenizer.batch_decode(base_generation, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("==="*20, "Base generation", "==="*20)
    print(base_model_generation[0])

base_output = model.model(base_generation)
prob = F.sigmoid(base_output.watermark_prob).detach().cpu().numpy().reshape((-1))
print(prob)
print(prob[input_ids.shape[-1]:].mean())


watermarked_generation = model.generate(
    input_ids,
    max_length=512,
    # attention_mask=attention_mask,
    num_beams=5,
    no_repeat_ngram_size=4,
    num_return_sequences=1,
    do_sample = True,
    top_p = 0.8,
    pad_token_id=tokenizer.pad_token_id,
    early_stopping=True,
)
watermarked_model_generation = tokenizer.batch_decode(watermarked_generation, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("==="*20, "Watermarked generation", "==="*20)
print(watermarked_model_generation[0])

watermarked_output = model.model(watermarked_generation)
prob = F.sigmoid(watermarked_output.watermark_prob).detach().cpu().numpy().reshape((-1))
print(prob)
print(prob[input_ids.shape[-1]:].mean())
