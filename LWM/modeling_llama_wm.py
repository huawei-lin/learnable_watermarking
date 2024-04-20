import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from dataclasses import dataclass, field
from transformers.utils import ModelOutput
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel
from peft.peft_model import PeftModel
from contextlib import contextmanager

IGNORE_INDEX = -100

@dataclass
class WatermarkCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    generator_loss: Optional[torch.FloatTensor] = None
    discriminator_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    watermark_prob: Optional[torch.FloatTensor] = None
    discriminator_label: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class Discriminator(nn.Module):
    def __init__(self, hidden_sizes):
        super().__init__()
        hidden_size_list = None
        if isinstance(hidden_sizes, int):
            hidden_size_list = [hidden_sizes, 1]
        else:
            hidden_size_list = hidden_sizes + [1]

        '''
        Name the discriminator layer with 'lora_' to pretend to be a lora adapter
        Peft lora will set all the module which name with 'lora_' to trainable
        https://github.com/huggingface/peft/blob/5a4b9cade64bac8afdff5006ee9dd815c90b5469/src/peft/tuners/lora/model.py#L253
        '''
        self.lora_layer_list = nn.ModuleList([nn.Linear(hidden_size_list[i],
            hidden_size_list[i + 1]) for i in range(len(hidden_size_list) - 1)])

    def forward(self, hidden_states):
        for layer in self.lora_layer_list:
            hidden_states = layer(hidden_states)
        return hidden_states


class WMLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        print(config)

        self.discriminator = Discriminator([config.hidden_size, 512, 32])
        self.post_init()

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

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
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
        logits = self.lm_head(hidden_states.clone())
        watermark_prob = self.discriminator(hidden_states.clone())


        loss = None
#         if labels is not None:
#             # Shift so that tokens < n predict n
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             # Flatten the tokens
#             loss_fct = CrossEntropyLoss()
#             shift_logits = shift_logits.view(-1, self.config.vocab_size)
#             shift_labels = shift_labels.view(-1)
#             # Enable model parallelism
#             shift_labels = shift_labels.to(shift_logits.device)
#             loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        

        return WatermarkCausalLMOutputWithPast(
            # loss=loss,
            logits=logits,
            watermark_prob=watermark_prob,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class AllInOneModel(nn.Module):
    def __init__(self, model, tokenizer, loss_alpha=0.5):
        nn.Module.__init__(self)
        self.loss_alpha = loss_alpha
        self.model = model
        self.tokenizer = tokenizer

        self.discriminator_stage = True
        self.generator_stage = True

        self.discriminator_num = 5
        self.generator_num = 2

        self.total_cnt = 0

    def get_stage(self):
        '''
        1. Train discriminator for num times
        2. Train discriminator and generator for num times
        '''
        if self.total_cnt < self.discriminator_num:
            self.discriminator_stage = True
            self.generator_stage = False
            self.total_cnt += 1
        elif self.total_cnt < self.discriminator_num + self.generator_num:
            self.discriminator_stage = True
            self.generator_stage = True
            self.total_cnt += 1

        if self.total_cnt >= self.discriminator_num + self.generator_num:
            self.total_cnt = 0

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

        self.get_stage()
        generator_loss = None
        discriminator_loss = None
        discriminator_loss_func = BCEWithLogitsLoss()

        print("***"*20)
        with self.model.disable_adapter():
            base_generation = self.model.generate(
                input_ids,
                max_length=512,
                attention_mask=attention_mask,
                num_beams=5,
                no_repeat_ngram_size=4,
                num_return_sequences=1,
                do_sample = True,
                top_p = 0.8,
                pad_token_id=self.tokenizer.pad_token_id,
                # early_stopping=True,
            )
            result = self.tokenizer.batch_decode(base_generation[:1,:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            print("base model result:", result)

        attention_mask_for_wm_training = base_generation.ne(self.tokenizer.pad_token_id)

        labels = base_generation.detach().clone()
        labels[:, :input_ids.shape[-1]] = IGNORE_INDEX
        labels[base_generation == self.tokenizer.pad_token_id] = IGNORE_INDEX


        if self.generator_stage:
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
            logits = outputs.logits
    
            result = torch.argmax(logits[:1], axis=2)
            result = self.tokenizer.batch_decode(result[:1,:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            print("lora model result:", result)
    
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                generator_loss = loss_fct(shift_logits, shift_labels)

        if self.discriminator_stage:
            wm_generation = self.model.generate(
                input_ids,
                max_length=512,
                attention_mask=attention_mask,
                num_beams=5,
                no_repeat_ngram_size=4,
                num_return_sequences=1,
                do_sample = True,
                top_p = 0.8,
                pad_token_id=self.tokenizer.pad_token_id,
                early_stopping=True,
            )
            result = self.tokenizer.batch_decode(wm_generation[:1,:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            print("watermarked model result:", result)
            attention_mask_of_wm_generation=wm_generation.ne(self.tokenizer.pad_token_id)
    
            with self.model.disable_adapter():
                diff = base_generation.shape[-1] - wm_generation.shape[-1]
        
                base_generation = F.pad(base_generation, (0, -diff if diff < 0 else 0, 0, 0), value=self.tokenizer.pad_token_id)
                wm_generation = F.pad(wm_generation, (0, diff if diff > 0 else 0, 0, 0), value=self.tokenizer.pad_token_id)
                attention_mask_for_wm_training = F.pad(attention_mask_for_wm_training, (0, -diff if diff < 0 else 0, 0, 0), value=False)
                attention_mask_of_wm_generation = F.pad(attention_mask_of_wm_generation, (0, diff if diff > 0 else 0, 0, 0), value=False)
        
                all_generation = torch.cat((base_generation, wm_generation))
                all_attn_mask = torch.cat((attention_mask_for_wm_training, attention_mask_of_wm_generation))
        
                all_generation_outputs = self.model(
                    input_ids=all_generation,
                    attention_mask=all_attn_mask,
                )
        
                base_label = torch.ones_like(base_generation)
                base_label[attention_mask_for_wm_training == False] = IGNORE_INDEX
                base_label[:, int(base_label.shape[-1]*0.1):] = IGNORE_INDEX
                base_label = base_label.flatten()
                wm_label = torch.zeros_like(wm_generation)
                wm_label[attention_mask_of_wm_generation == False] = IGNORE_INDEX
                wm_label[:, int(wm_label.shape[-1]*0.1):] = IGNORE_INDEX
                wm_label = wm_label.flatten()
                discriminator_label = torch.cat((base_label, wm_label)).float()
        
                watermark_prob = all_generation_outputs.watermark_prob.flatten()
        
                watermark_prob = watermark_prob[discriminator_label != IGNORE_INDEX]
                discriminator_label = discriminator_label[discriminator_label != IGNORE_INDEX]
        
                discriminator_loss = discriminator_loss_func(watermark_prob, discriminator_label.float())

        if self.generator_stage and self.discriminator_stage:
            loss = self.loss_alpha*generator_loss + (1 - self.loss_alpha)*discriminator_loss
        elif self.generator_stage:
            loss = generator_loss
        elif self.discriminator_stage:
            loss = discriminator_loss
        else:
            raise ValueError(
                "generator_stage and discriminator_stage can not both be"
                " false at the same time."
            )

        return WatermarkCausalLMOutputWithPast(
            loss=loss,
            generator_loss=generator_loss,
            discriminator_loss=discriminator_loss,
            watermark_prob=watermark_prob,
            discriminator_label=discriminator_label,
        )
