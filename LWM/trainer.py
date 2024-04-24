import os
import transformers
from transformers import Trainer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.integrations.integration_utils import WandbCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import wandb
import torch


class WatermarkTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.outputs = None

    def log(self, logs: Dict[str, float]) -> None:
        def get_item(x):
            return x.item() if x is not None else 0


        generator_loss = get_item(self.outputs.generator_loss)
        discriminator_loss = get_item(self.outputs.discriminator_loss)
        discriminator_acc = get_item(self.outputs.discriminator_acc)
        stage = -1*int(self.model.discriminator_stage) + int(self.model.generator_stage)
        loss_alpha = self.model.loss_alpha

        if self.is_world_process_zero():
            text_table = wandb.Table(columns=[
                "generator_loss",
                "discriminator_loss",
                "discriminator_acc",
                "base_model_generation",
                "watermarked_generation",
            ])
            for idx, (base_model_generation, watermarked_generation) \
                in enumerate(zip(self.outputs.base_model_generation, self.outputs.watermarked_generation)):
        
                text_table.add_data(generator_loss, discriminator_loss, discriminator_acc, \
                    base_model_generation, watermarked_generation)
    
            for callback in self.callback_handler.callbacks:
                if isinstance(callback, WandbCallback):
                    callback._wandb.log({"samples_vis": text_table}, commit=False)

        logs = {**logs, **{
            "generator_loss": generator_loss,
            "discriminator_loss": discriminator_loss,
            "discriminator_acc": discriminator_acc,
            "stage": stage,
            "loss_alpha": loss_alpha,
        }}
        super().log(logs)


    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        self.outputs = outputs
        return (loss, outputs) if return_outputs else loss

    def _save_checkpoint(self, model, trial, metrics=None):
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)

        DISCRIMINATOR_WEIGHTS_NAME = "discriminator.bin"
        unwrapped_model = self.accelerator.unwrap_model(model)
        if self.args.should_save:
            state_dict = unwrapped_model.model.discriminator.state_dict()
            torch.save(state_dict, os.path.join(output_dir, DISCRIMINATOR_WEIGHTS_NAME))

        return super()._save_checkpoint(unwrapped_model.model.peft_model, trial, metrics)


