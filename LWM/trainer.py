import transformers
from transformers import Trainer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch


class WatermarkTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.outputs = None

    def log(self, logs: Dict[str, float]) -> None:
        def get_acc(pred, label):
            pred[pred >= 0] = 1
            pred[pred < 0] = 0
            return torch.sum(pred == label)/len(pred)
        logs = {**logs, **{
            "generator_loss": self.outputs.generator_loss.item(),
            "discriminator_loss": self.outputs.discriminator_loss.item(),
            "discriminator_acc": get_acc(self.outputs.watermark_prob, self.outputs.discriminator_label).item(),
        }}
        super().log(logs)



    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        self.outputs = outputs
        return (loss, outputs) if return_outputs else loss

