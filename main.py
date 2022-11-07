from datetime import datetime
from typing import Optional

import pytorch_lightning as pl
from loguru import logger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from transformers import ViltForMaskedLM, ViltProcessor
from typer import Option, Typer

from app.dataset import ViltDataModule
from app.module import ViltModule

cmd = Typer(pretty_exceptions_show_locals=False)


@cmd.command(no_args_is_help=True)
def train(
    optimizer: str = Option("adamw", help="optimizer name", rich_help_panel="model"),
    learning_rate: float = Option(5e-4, help="learning rate", rich_help_panel="model"),
    weigth_decay: float = Option(1e-4, help="weight decay", rich_help_panel="model"),
    batch_size: int = Option(32, min=1, help="batch size", rich_help_panel="data"),
    num_workers: int = Option(
        0, min=0, help="num workers for dataloader", rich_help_panel="train"
    ),
    accumulate_grad_batches: Optional[int] = Option(
        None, min=1, help="accumulate grad batches", rich_help_panel="train"
    ),
    gradient_clip_val: Optional[float] = Option(
        None, min=0.0, help="gradient clip value", rich_help_panel="train"
    ),
    auto_scale_batch_size: bool = Option(
        False,
        help="auto find batch size, ignore batch_size option",
        rich_help_panel="train",
    ),
    max_epochs: int = Option(3, help="max epochs", rich_help_panel="train"),
    steps_per_epoch: Optional[int] = Option(
        None, min=1, help="steps per epoch", rich_help_panel="train"
    ),
    fast_dev_run: bool = Option(False, help="do test run", rich_help_panel="train"),
    save_path: str = Option(
        "save/my_model", help="save path of trained model", rich_help_panel="train"
    ),
    log_every_n_steps: int = Option(
        100, help="log every n steps", rich_help_panel="train"
    ),
    resume_from_checkpoint: Optional[str] = Option(
        None,
        help="Path/URL of the checkpoint from which training is resumed",
        rich_help_panel="train",
    ),
    wandb_name: Optional[str] = Option(
        None, help="wandb project name", rich_help_panel="train"
    ),
    seed: Optional[int] = Option(None, help="seed", rich_help_panel="train"),
):
    logger.debug("loading transformers model, processor")
    model = ViltForMaskedLM.from_pretrained("Bingsu/my_vilt_b32_mlm_not_trained")
    processor = ViltProcessor.from_pretrained("Bingsu/my_vilt_b32_mlm_not_trained")

    logger.debug("loading datamodule")
    datamodule = ViltDataModule(
        processor=processor, batch_size=batch_size, num_workers=num_workers
    )

    logger.debug("loading lightning module")
    module = ViltModule(
        model=model,
        processor=processor,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weigth_decay,
    )
    checkpoints = ModelCheckpoint(
        monitor="train/loss_epoch",
        save_last=True,
        filename="{epoch:02d}-{train/loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    callbacks = [checkpoints, RichProgressBar(), LearningRateMonitor("step")]

    if seed is not None:
        pl.seed_everything(seed)
        logger.info(f"set seed: {seed}")

    limit_train_batches = steps_per_epoch if steps_per_epoch else 1.0
    if isinstance(limit_train_batches, int) and accumulate_grad_batches is not None:
        limit_train_batches *= accumulate_grad_batches

    if not wandb_name:
        now = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
        wandb_name = f"vilt_{now}"
    logger.info(f"wandb name: {wandb_name}")

    logger.debug("set trainer")
    trainer = pl.Trainer(
        logger=WandbLogger(name=wandb_name, project="vilt_mlm"),
        fast_dev_run=fast_dev_run,
        enable_progress_bar=True,
        accelerator="auto",
        precision=16 if "bnb" not in optimizer else 32,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        callbacks=callbacks,
        auto_scale_batch_size=auto_scale_batch_size,
        log_every_n_steps=log_every_n_steps,
    )

    if auto_scale_batch_size:
        logger.debug("auto scale batch size")
        trainer.tune(module, datamodule=datamodule)

    logger.debug("start training")
    trainer.fit(module, datamodule=datamodule, ckpt_path=resume_from_checkpoint)
    logger.debug("training finished")

    module.save(save_path)
    logger.info(f"model saved at: {save_path}")


if __name__ == "__main__":
    cmd()
