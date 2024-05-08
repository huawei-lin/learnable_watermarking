import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import json
import copy
import logging
import random
random.seed(42)

import transformers
import torch
torch.manual_seed(42)
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from datasets import load_dataset
from evaluate import load

import numpy as np
np.random.seed(42)
# np.set_printoptions(precision=2)

from LWM.loaders import get_model_tokenizer
from LWM.evaluate import Perplexity
from LWM.utils import (
    ModelArguments,
    TrainingArguments,
    AdapterArguments
)
from LWM.openai_utils import openai_completion, OpenAIDecodingArguments, init_openai

# os.environ["OPENAI_ORG"] = "your org key"
# os.environ["OPENAI_API_KEY"] = "your api key"

max_model_length = 512

init_openai()
decoding_args = OpenAIDecodingArguments()
decoding_args.max_tokens = max_model_length

dataset = load_dataset("allenai/c4", "realnewslike", split="validation")

base_model_path = "meta-llama/Meta-Llama-3-8B"

evaluate_model = "google/gemma-7b"
perplexity = Perplexity(evaluate_model)

checkpoints_path = "/home/hl3352/LLMs/LearnableWatermarking/learnable_watermarking/exp_wikitext/stage_limit_decay0.99_wiki_t16_3L_dense_alpha0.5_ne1_llama3_8b_qv_r8_a16_lr1e-4_bs0/checkpoint-1440"
checkpoints_path = "/home/hl3352/LLMs/LearnableWatermarking/learnable_watermarking/exp_wikitext/c60_75T90_decay0.99_wiki_t16_3L_dense_alpha0.5_ne1_llama3_8b_qv_r8_a16_lr1e-4_bs0/checkpoint-750"
# checkpoints_path = "/home/hl3352/LLMs/LearnableWatermarking/learnable_watermarking/exp_wikitext/0.1_stage_limit_decay0.99_wiki_t16_3L_dense_alpha0.5_ne1_llama3_8b_qv_r8_a16_lr1e-4_bs0/checkpoint-390"

openai_attack_prompt = "As an expert copy-editor, please rewrite the following text in your own voice while ensuring that the final output contains the same information as the original text and has roughly the same length. Please paraphrase all sentences and do not omit any crucial details. Additionally, please take care to provide any relevant information about public figures, organizations, or other entities mentioned in the text to avoid any potential misunderstandings or biases."
# openai_attack_prompt = "paraphrase the following paragraphs:\n"

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

def weighted_arithmetic_mean(x):
    x = x[...,:-1] # the last element can be an outlier
    w = np.linspace(0, 1, max_model_length)[:x.shape[-1]]
    return np.dot(x, w)/w.sum()

result_path = "./evaluation/ppls/eval_result.jsonl"
with open(result_path, "w") as fw:
    pass

generate_args = {
    "num_beams": 5,
    "no_repeat_ngram_size": 4,
    "num_return_sequences": 1,
    "do_sample":  True,
    "top_p": 0.8,
    "pad_token_id": tokenizer.pad_token_id,
    "early_stopping": True,
}


result_list = []
for i, prompt in enumerate(dataset['text']):
    total_result = {}
    label_ids = tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                max_length=max_model_length,
                truncation=True,
            ).input_ids

    prompt = " ".join(prompt.split(" ")[:10])

    total_result["prompt"] = prompt

    tokenized = tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                max_length=max_model_length,
                truncation=True,
            )
    input_ids = tokenized.input_ids.cuda()
    
    openai_generation = openai_completion(prompt, decoding_args, model_name="gpt-3.5-turbo-0125")
    openai_generation = prompt + " " + openai_generation
    openai_ids = tokenizer(
        openai_generation,
        return_tensors="pt",
        padding="longest",
        max_length=max_model_length,
        truncation=True,
    ).input_ids.cuda()
    openai_output = model.model(openai_ids)
    prob = F.sigmoid(openai_output.watermark_prob).detach().cpu().numpy().reshape((-1))
    openai_prob = weighted_arithmetic_mean(prob[input_ids.shape[-1]:])
    openai_ppls = perplexity(openai_generation)['ppls']

    openai_result = {
        "generation": openai_generation,
        "prob": openai_prob,
        "prob_list": prob.tolist(),
        "ppls": openai_ppls,
    }
    total_result["openai_result"] = openai_result

    
    with model.disable_adapter():
        base_generation = model.generate(
            input_ids,
            max_length=max_model_length,
            **generate_args
        )
        base_model_generation = tokenizer.batch_decode(base_generation, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    base_output = model.model(base_generation)
    prob = F.sigmoid(base_output.watermark_prob).detach().cpu().numpy().reshape((-1))
    base_prob = weighted_arithmetic_mean(prob[input_ids.shape[-1]:])
    base_ppls = perplexity(base_model_generation)['ppls']
    base_result = {
        "generation": base_model_generation,
        "prob": base_prob,
        "prob_list": prob.tolist(),
        "ppls": base_ppls,
    }
    total_result["base_result"] = base_result 
    
    
    watermarked_generation = model.generate(
        input_ids,
        max_length=max_model_length,
        **generate_args
    )
    watermarked_model_generation = tokenizer.batch_decode(watermarked_generation, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    watermarked_output = model.model(watermarked_generation)
    prob = F.sigmoid(watermarked_output.watermark_prob).detach().cpu().numpy().reshape((-1))
    watermarked_prob = weighted_arithmetic_mean(prob[input_ids.shape[-1]:])
    watermarked_ppls = perplexity(watermarked_model_generation)['ppls']
    watermarked_result = {
        "generation": watermarked_model_generation,
        "prob": watermarked_prob,
        "prob_list": prob.tolist(),
        "ppls": watermarked_ppls,
    }
    total_result["watermarked_result"] = watermarked_result

    openai_attack_generation = openai_completion(f"{openai_attack_prompt}: \"{watermarked_model_generation}\"", decoding_args, model_name="gpt-3.5-turbo")
    openai_attack_ids = tokenizer(
        openai_attack_generation,
        return_tensors="pt",
        padding="longest",
        max_length=max_model_length,
        truncation=True,
    ).input_ids.cuda()
    openai_attack_output = model.model(openai_attack_ids)
    prob = F.sigmoid(openai_attack_output.watermark_prob).detach().cpu().numpy().reshape((-1))
    openai_attack_prob = weighted_arithmetic_mean(prob[input_ids.shape[-1]:])
    openai_attack_ppls = perplexity(openai_attack_generation)['ppls']

    openai_attack_result = {
        "generation": openai_attack_generation,
        "prob": openai_attack_prob,
        "prob_list": prob.tolist(),
        "ppls": openai_attack_ppls,
    }
    total_result["openai_attack_result"] = openai_attack_result

    print("==="*20)
    print(i, "prob:", {k: v["prob"] for k, v in total_result.items() if isinstance(v, dict) and "prob" in v.keys()})
    print(i, "ppls:", {k: v["ppls"] for k, v in total_result.items() if isinstance(v, dict) and "ppls" in v.keys()})
    # result_list.append(total_result)
    with open(result_path, "a") as fw:
        fw.write(json.dumps(total_result) + "\n")
