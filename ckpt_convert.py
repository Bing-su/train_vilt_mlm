from loguru import logger
from typer import Argument, Option, Typer

from app.module import ViltModule

cli = Typer()


@cli.command(no_args_is_help=True)
def convert(
    ckpt_path: str = Argument(..., help="checkpoint path"),
    save_path: str = Option("save/converted", help="save path of converted model"),
):
    module = ViltModule.load_from_checkpoint(ckpt_path)
    logger.debug(f"checkpoint loaded from {ckpt_path}")
    module.save(save_path)
    logger.debug(f"model saved to {save_path}")


if __name__ == "__main__":
    cli()
