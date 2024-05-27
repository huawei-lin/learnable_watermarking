import os
import time
import math
import transformers
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.configuration_utils import PretrainedConfig
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from dataclasses import dataclass, field
from transformers.utils import ModelOutput
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaModel,
    LlamaDecoderLayer,
    LlamaRMSNorm
)
from peft.peft_model import PeftModel
from peft.utils.other import _get_submodules
from peft.tuners.tuners_utils import BaseTunerLayer
from contextlib import contextmanager, nullcontext
from .evaluate import Timer
from copy import copy

IGNORE_INDEX = -100

generate_kwargs = {
    "num_beams": 4,
    "no_repeat_ngram_size": 4,
    "num_return_sequences": 1,
    "do_sample": True,
    "top_p": 0.8,
    "early_stopping":True,
}


transformers_output = {}
def get_transformers_ouput(i):
    def hook(model, input, output):
        transformers_output[i] = output[0]
    return hook

transformers_input_kwargs = {}
def get_transformers_input(i):
    def hook(model, input, kwargs):
        transformers_input_kwargs[i] = kwargs
    return hook


@dataclass
class WatermarkCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    gen_dis_pred: Optional[torch.FloatTensor] = None
    watermark_prob: Optional[torch.FloatTensor] = None
    pred_index_offset: Optional[List[int]] = None
    discriminator_pred: Optional[torch.FloatTensor] = None # binarize watermark_prob
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    base_model_generation: Optional[Tuple[str]] = None
    watermarked_generation: Optional[Tuple[str]] = None
    logs: Optional[Dict[str, Any]] = None


@dataclass
class WatermarkConfig():
    pass


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layers_num = 3
        hidden_size_list = [1024, 128, 1]

        self.decoder_layer_list = nn.ModuleList([
            LlamaDecoderLayer(config, config.num_hidden_layers + i) for i in range(self.layers_num)
        ])

        hidden_size_list = [config.hidden_size, *hidden_size_list]
        self.mlp_layer_list = nn.ModuleList()
        for i in range(len(hidden_size_list) - 1):
            self.mlp_layer_list.append(nn.Linear(hidden_size_list[i], hidden_size_list[i + 1]))
            if hidden_size_list[i + 1] != 1:
                self.mlp_layer_list.append(nn.SiLU())

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        for layer in self.decoder_layer_list:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            hidden_states = layer_outputs[0]

        for layer in self.mlp_layer_list:
            hidden_states = layer(hidden_states)

        return hidden_states


class WatermarkedLlama(nn.Module):
    def __init__(self, peft_model, watermark_config: Optional[WatermarkConfig] = None):
        nn.Module.__init__(self)

        self.watermark_config = watermark_config
        self.peft_model = peft_model

        ###########
#         for name, module in self.peft_model.named_modules():
#             if 'lora_A.default' in name or 'lora_B.default' in name:
#                 # nn.init.normal_(module.weight, std=1/min(module.weight.shape))
#                 nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        ###########

        self.disc_layernorm = LlamaRMSNorm(peft_model.config.hidden_size, eps=peft_model.config.rms_norm_eps)
        self.discriminator = Discriminator(peft_model.config)

        # Copy the last decoder layer to discriminator
        for i in range(self.discriminator.layers_num):
            self.replace_transformer_weight(self.get_decoder().layers[self.discriminator.layers_num - 1], self.discriminator.decoder_layer_list[i])

        for i, layer in enumerate(self.get_decoder().layers):
            layer.register_forward_hook(get_transformers_ouput(i))
        self.get_decoder().layers[0].register_forward_pre_hook(get_transformers_input(0), with_kwargs=True)

    def replace_transformer_weight(self, source_module, target_module):
        layer_idx = source_module.self_attn.layer_idx
        target_module.self_attn.layer_idx = layer_idx
        key_list = [key for key, _ in target_module.named_modules()]
        for key in key_list:
            if "." not in key or "rotary_emb" in key or "act_fn" in key:
                continue
            _, source_layer, _ = _get_submodules(source_module, key)
            _, target_layer, _ = _get_submodules(target_module, key)
            if isinstance(source_layer, BaseTunerLayer):
                source_layer = source_layer.base_layer
            target_layer.weight = nn.Parameter(source_layer.weight.detach().clone())

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.peft_model, name)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, WatermarkCausalLMOutputWithPast]:

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.get_decoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]

        logits = self.get_output_embeddings()(hidden_states.clone())

        transformers_output_sum = torch.stack(tuple(transformers_output.values())).sum(axis=0)

        watermark_prob = self.discriminator(
            self.disc_layernorm(transformers_output_sum),
            **transformers_input_kwargs[0],
        )

        return WatermarkCausalLMOutputWithPast(
            logits=logits,
            watermark_prob=watermark_prob,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class AllInOneModel(LlamaForCausalLM, nn.Module):
    def __init__(self, model, tokenizer, loss_alpha=0.5):
        nn.Module.__init__(self)
        self.loss_alpha = loss_alpha
        self.loss_alpha_diff = loss_alpha
        self.model = model
        self.tokenizer = tokenizer

        self.discriminator_stage = True
        self.generator_stage = False

        self.discriminator_num = 40
        self.generator_num = 5
        self.first_loop = True

        self.total_cnt = 0
        self.stage_cnt = 0

        self.queue_len = 16
        self.queue_idx = 0
        self.acc_queue = torch.zeros((self.queue_len))

        self.timer = Timer()

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

#     def __call__(self, *args, **kwargs):
#         return self.model.__call__(*args, **kwargs)

    @contextmanager
    def mark_adapters_as_untrainable(self):
        name2trainable = {}
        for n, p in self.model.named_parameters():
            name2trainable[n] = p.requires_grad
        try:
            for n, p in self.model.named_parameters():
                if "lora_" in n and "discriminator" not in n:
                    p.requires_grad = False
            yield
        finally:
            for n, p in self.model.named_parameters():
                p.requires_grad = name2trainable[n]

    @contextmanager
    def mark_discriminator_as_untrainable(self):
        name2trainable = {}
        for n, p in self.model.named_parameters():
            name2trainable[n] = p.requires_grad
        try:
            for n, p in self.model.named_parameters():
                if "discriminator" in n:
                    p.requires_grad = False
            yield
        finally:
            for n, p in self.model.named_parameters():
                p.requires_grad = name2trainable[n]

    def get_stage(self):
        self.acc_queue_mean = self.acc_queue.mean()

        if self.discriminator_stage == True and self.acc_queue_mean > 0.85:
            self.discriminator_stage = False
            self.generator_stage = True
            self.stage_cnt = 0

        elif self.generator_stage == True and (self.acc_queue_mean > 0.85 or self.acc_queue_mean < 0.75):
        # elif self.generator_stage == True and self.acc_queue_mean < 0.75:
            self.discriminator_stage = True
            self.generator_stage = False

            self.total_cnt += 1
            self.stage_cnt = 0
            self.first_loop = False

        self.stage_cnt += 1
        if self.generator_stage == True:
            self.loss_alpha_diff = self.loss_alpha_diff*0.995
            self.loss_alpha = 1 - self.loss_alpha_diff

    def get_acc(self, pred, label):
        if pred is None or label is None:
            return None
        label[label >= 0.5] = 1
        label[label < 0.5] = 0
        return torch.sum(pred == label)/len(pred)

    def get_generate_kwargs(self):
        kwargs = copy(generate_kwargs)
        kwargs["num_beams"] = random.randint(1, 5)
        kwargs["no_repeat_ngram_size"] = random.randint(1, 5)
        return kwargs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        self.timer.start()

        generator_loss = None
        discriminator_loss = None
        discriminator_loss_func = BCEWithLogitsLoss()

        batch_size, input_length = input_ids.shape
        self.get_stage()

        self.timer.get_time_and_restart("get_stage")

        with self.model.disable_adapter():
            base_generation = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                max_length=self.tokenizer.model_max_length,
                **self.get_generate_kwargs(),
            )
            base_model_generation = self.tokenizer.batch_decode(base_generation, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            base_generation_len = base_generation.shape[-1]

        self.timer.get_time_and_restart("get_base_model_generation")

        attention_mask_for_wm_training = base_generation.ne(self.tokenizer.pad_token_id)

        labels = base_generation.detach().clone()
        labels[:, :input_ids.shape[-1]] = IGNORE_INDEX
        labels[base_generation == self.tokenizer.pad_token_id] = IGNORE_INDEX

        self.timer.get_time_and_restart("construct_label_for_base_generation")

        with self.mark_discriminator_as_untrainable() if self.generator_stage else torch.no_grad():
            outputs = self.model(
                input_ids=base_generation,
                attention_mask=attention_mask_for_wm_training,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
            self.timer.get_time_and_restart("generator forwarding")
            discriminator_label = torch.ones_like(base_generation) - torch.rand(base_generation.shape, device=base_generation.device)/10
            # discriminator_label[attention_mask_for_wm_training == False] = IGNORE_INDEX
            # discriminator_label[:, :int(discriminator_label.shape[-1]*0.1)] = IGNORE_INDEX
            discriminator_label[attention_mask_for_wm_training == False] = IGNORE_INDEX
            discriminator_label[:, :input_length] = IGNORE_INDEX
            discriminator_label = discriminator_label.flatten()

            watermark_prob = outputs.watermark_prob.flatten()

            watermark_prob = watermark_prob[discriminator_label != IGNORE_INDEX]
            discriminator_label = discriminator_label[discriminator_label != IGNORE_INDEX]

            gen_dis_pred = watermark_prob.detach().clone()
            gen_dis_pred[gen_dis_pred >= 0] = 1
            gen_dis_pred[gen_dis_pred < 0] = 0

            gen_discriminator_loss = discriminator_loss_func(watermark_prob, discriminator_label.float())
            gen_discriminator_acc = self.get_acc(gen_dis_pred, discriminator_label.detach().clone())

            logits = outputs.logits

            result = torch.argmax(logits, axis=2)
            result = self.tokenizer.batch_decode(result, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            generation_loss = loss_fct(shift_logits, shift_labels)

            generator_loss = generation_loss

        self.timer.get_time_and_restart("generator loss calculating")

        wm_generation = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=self.tokenizer.model_max_length,
            **self.get_generate_kwargs(),
        )
        watermarked_generation = self.tokenizer.batch_decode(wm_generation, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        attention_mask_of_wm_generation=wm_generation.ne(self.tokenizer.pad_token_id)
        wm_generation_len = wm_generation.shape[-1]
        self.timer.get_time_and_restart("watermarked_generation")

        pred_index_offset = [(x[input_length:] == True).sum().item() for atten_masks in (attention_mask_for_wm_training, attention_mask_of_wm_generation) for x in atten_masks]

        self.timer.get_time_and_restart("get pred_index_offset")
        # with self.mark_adapters_as_untrainable() if self.discriminator_stage else torch.no_grad():
        with self.mark_adapters_as_untrainable() if self.discriminator_stage else nullcontext():
            diff = base_generation.shape[-1] - wm_generation.shape[-1]

            base_generation = F.pad(base_generation, (0, -diff if diff < 0 else 0, 0, 0), value=self.tokenizer.pad_token_id)
            wm_generation = F.pad(wm_generation, (0, diff if diff > 0 else 0, 0, 0), value=self.tokenizer.pad_token_id)
            attention_mask_for_wm_training = F.pad(attention_mask_for_wm_training, (0, -diff if diff < 0 else 0, 0, 0), value=False)
            attention_mask_of_wm_generation = F.pad(attention_mask_of_wm_generation, (0, diff if diff > 0 else 0, 0, 0), value=False)


            all_generation = torch.cat((base_generation, wm_generation))
            all_attn_mask = torch.cat((attention_mask_for_wm_training, attention_mask_of_wm_generation))

            self.timer.get_time_and_restart("construct mask")

            all_generation_outputs = self.model(
                input_ids=all_generation,
                attention_mask=all_attn_mask,
            )

            self.timer.get_time_and_restart("discriminator forwarding")

            base_label = torch.zeros_like(base_generation) + torch.rand(base_generation.shape, device=base_generation.device)/10
            base_label[attention_mask_for_wm_training == False] = IGNORE_INDEX
            # base_label[:, :int(base_label.shape[-1]*0.1)] = IGNORE_INDEX
            base_label[:, :input_length] = IGNORE_INDEX
            base_label = base_label.flatten()
            wm_label = torch.ones_like(wm_generation) - torch.rand(wm_generation.shape, device=wm_generation.device)/10
            wm_label[attention_mask_of_wm_generation == False] = IGNORE_INDEX
            # wm_label[:, :int(wm_label.shape[-1]*0.1)] = IGNORE_INDEX
            wm_label[:, :input_length] = IGNORE_INDEX
            wm_label = wm_label.flatten()
            discriminator_label = torch.cat((base_label, wm_label))

            watermark_prob = all_generation_outputs.watermark_prob.flatten()

            watermark_prob = watermark_prob[discriminator_label != IGNORE_INDEX]
            discriminator_label = discriminator_label[discriminator_label != IGNORE_INDEX]

            discriminator_loss = discriminator_loss_func(watermark_prob, discriminator_label.float())

            pred = watermark_prob.detach().clone()
            pred[pred >= 0] = 1
            pred[pred < 0] = 0

            discriminator_acc = self.get_acc(pred, discriminator_label.detach().clone())

            self.acc_queue[self.queue_idx] = discriminator_acc
            self.queue_idx = (self.queue_idx + 1)%self.queue_len

            self.timer.get_time_and_restart("discriminator loss calculating")

        if self.generator_stage:
            loss = self.loss_alpha*generator_loss + (1 - self.loss_alpha)*discriminator_loss
        elif self.discriminator_stage:
            loss = discriminator_loss
        else:
            raise ValueError(
                "generator_stage and discriminator_stage can not both be"
                " false at the same time."
            )

        logs = {
            "generator_loss": generator_loss.item(),
            "generation_loss": generation_loss.item(),
            "base_generation_len": base_generation_len,
            "watermarked_generation_len": wm_generation_len,
            "gen_discriminator_loss": gen_discriminator_loss.item(),
            "discriminator_loss": discriminator_loss.item(),
            "discriminator_acc": discriminator_acc.item(),
            "gen_discriminator_acc": gen_discriminator_acc.item(),
            "loss_alpha": self.loss_alpha,
            "stage": -1*int(self.discriminator_stage) + int(self.generator_stage),
            "acc_queue_mean": self.acc_queue_mean.item(),
        }

        self.timer.get_time_and_restart("finalizing")

        print(self.timer.get_record())

        return WatermarkCausalLMOutputWithPast(
            loss=loss,
            gen_dis_pred=gen_dis_pred,
            discriminator_pred=pred,
            base_model_generation=base_model_generation,
            watermarked_generation=watermarked_generation,
            pred_index_offset=pred_index_offset,
            logs=logs,
        )
