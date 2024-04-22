import transformers
from transformers import Trainer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.integrations.integration_utils import WandbCallback
import wandb
import torch


class WatermarkTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.outputs = None

    def log(self, logs: Dict[str, float]) -> None:
        def get_item(x):
            return x.item() if x is not None else 0

        def get_acc(pred, label):
            if pred is None or label is None:
                return None
            pred[pred >= 0] = 1
            pred[pred < 0] = 0
            label[label >= 0.5] = 1
            label[label < 0.5] = 0
            return torch.sum(pred == label)/len(pred)

        generator_loss = get_item(self.outputs.generator_loss)
        discriminator_loss = get_item(self.outputs.discriminator_loss)
        discriminator_acc = get_item(get_acc(self.outputs.watermark_prob, self.outputs.discriminator_label))
        stage = -1*int(self.model.discriminator_stage) + int(self.model.generator_stage)
        loss_alpha = self.model.loss_alpha

        text_table = wandb.Table(columns=[
            "generator_loss",
            "discriminator_loss",
            "discriminator_acc",
            "base_model_generation",
            "watermarked_generation",
        ])
        for base_model_generation, watermarked_generation \
            in zip(self.outputs.base_model_generation, self.outputs.watermarked_generation):

            text_table.add_data(generator_loss, discriminator_loss, discriminator_acc, \
                base_model_generation, watermarked_generation)

#         if self.is_world_process_zero():
#             for base_model_generation, watermarked_generation \
#                 in zip(self.outputs.base_model_generation, self.outputs.watermarked_generation):
#         
#                 text_table.add_data(generator_loss, discriminator_loss, discriminator_acc, \
#                     base_model_generation, watermarked_generation)
#     
#             for callback in self.callback_handler.callbacks:
#                 if isinstance(callback, WandbCallback):
#                     print("_initialized:", callback._initialized)
#                     callback._wandb.log({"samples_vis": text_table}, commit=False)

        logs = {**logs, **{
            "generator_loss": generator_loss,
            "discriminator_loss": discriminator_loss,
            "discriminator_acc": discriminator_acc,
            "stage": stage,
            "loss_alpha": loss_alpha,
            "samples_vis": text_table,
        }}
        super().log(logs)


    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        self.outputs = outputs
        return (loss, outputs) if return_outputs else loss

