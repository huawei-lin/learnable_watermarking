import transformers
from transformers import Trainer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
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
            print("pred:", pred)
            print("label:", label)
            return torch.sum(pred == label)/len(pred)

        logs = {**logs, **{
            "generator_loss": get_item(self.outputs.generator_loss),
            "discriminator_loss": get_item(self.outputs.discriminator_loss),
            "discriminator_acc": get_item(get_acc(self.outputs.watermark_prob, self.outputs.discriminator_label)),
            "stage": -1*int(self.model.discriminator_stage) + int(self.model.generator_stage),
            "loss_alpha": self.model.loss_alpha,
        }}
        super().log(logs)



    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        self.outputs = outputs
        return (loss, outputs) if return_outputs else loss

