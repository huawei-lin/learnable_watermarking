import os
import transformers
import numpy as np
from transformers import Trainer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.integrations.integration_utils import WandbCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import wandb
import torch
from transformers.utils import (
    logging,
    is_peft_available,
    is_safetensors_available,
    WEIGHTS_NAME,
    SAFE_WEIGHTS_NAME
)
from transformers.modeling_utils import PreTrainedModel

if is_safetensors_available():
    import safetensors.torch

if is_peft_available():
    from peft import PeftModel

logger = logging.get_logger(__name__)

TRAINING_ARGS_NAME = "training_args.bin"


class WatermarkTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.outputs = None

    def log(self, logs: Dict[str, float]) -> None:
        def get_item(x):
            if x is None:
                return 0
            if torch.is_tensor(x):
                return x.item()
            return x

        if self.is_world_process_zero():
            text_table = wandb.Table(columns=[
                "base_model_generation",
                "gen_dis_pred",
                "watermarked_generation",
                "discriminator_pred",
            ])
            with np.printoptions(threshold=np.inf):
                text_table.add_data(self.outputs.base_model_generation[0], str(self.outputs.gen_dis_pred.detach().cpu().numpy()), self.outputs.watermarked_generation[0], str(self.outputs.discriminator_pred.detach().cpu().numpy()))
#             for idx, (base_model_generation, watermarked_generation) \
#                 in enumerate(zip(self.outputs.base_model_generation, self.outputs.watermarked_generation)):
#         
#                 text_table.add_data(base_model_generation, watermarked_generation)
    
            for callback in self.callback_handler.callbacks:
                if isinstance(callback, WandbCallback):
                    callback._wandb.log({"samples_vis": text_table}, commit=False)

        logs = {**logs, **self.outputs.logs}
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

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        model_to_save = self.model.peft_model ### DIFF

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(model_to_save, supported_classes):
            if state_dict is None:
                state_dict = model_to_save.state_dict()

            if isinstance(unwrap_model(model_to_save), supported_classes):
                unwrap_model(model_to_save).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            model_to_save.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
