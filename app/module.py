import pytorch_lightning as pl
import torch
from loguru import logger
from transformers import ViltForMaskedLM, ViltProcessor

from .util import create_optimizer


class ViltModule(pl.LightningModule):
    def __init__(
        self,
        model: ViltForMaskedLM,
        processor: ViltProcessor,
        optimizer: str = "adamw",
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = model
        self.processor = processor

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    @property
    def forward(self):
        return self.model.forward

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        loss = output.loss

        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        params = self.model.named_parameters()

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        opt_class = create_optimizer(self.optimizer)
        optimizer = opt_class(
            optimizer_grouped_parameters,
            lr=5e-5,
        )

        if "bnb" in self.optimizer:
            from bitsandbytes.optim import GlobalOptimManager

            manager = GlobalOptimManager.get_instance()
            modules = self.model.modules()
            for module in modules:
                if isinstance(module, torch.nn.Embedding):
                    manager.register_module_override(
                        module, "weight", {"optim_bits": 32}
                    )
                    logger.debug(f"bitsandbytes: will optimize {module} in fp32")

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
        )

        scheduler_config = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_config]

    def save(self, save_path: str):
        self.model.save_pretrained(save_path)
        logger.info(f"model saved at '{save_path}'")
        self.processor.save_pretrained(save_path)
        logger.info(f"processor saved at '{save_path}'")
