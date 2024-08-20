import glob
import os
from pathlib import Path
import torch
import wandb
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from rich.console import Console
from aigpro.models.model_module import PDBModelModule

console = Console()
# set ENV

torch.set_float32_matmul_precision("high")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
CACHE_DIR = "/scratch/lab09/wandb_cache"
STAGING_DIR = "/scratch/lab09/wandb_stagging"
console = Console()

WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
if WANDB_API_KEY is None:
    console.print("WANDB_API_KEY is not set")
    exit(1)

os.environ["WANDB_BASE_URL"] = os.environ.get("WANDB_BASE_URL")  # type: ignore
os.environ["WANDB_MODE"] = "online"

try:
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_CACHE_DIR"] = CACHE_DIR
    console.print(f"Set WANDB_CACHE_DIR = {CACHE_DIR}")
    Path(STAGING_DIR).mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DATA_DIR"] = STAGING_DIR
    console.print(f"Set WANDB_DATA_DIR = {STAGING_DIR}")

except (FileExistsError, PermissionError) as e:
    console.print(f"Error creating directory: {e}")


def reset_weights(m) -> None:
    """Try resetting model weights to avoid.

    weight leakage.
    """
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            console.print(f"Reset trainable parameters of layer = {layer}")
            layer.reset_parameters()


def find_yaml_files(path: str) -> list:
    """Find all yaml files in a directory.

    Args:
        path (str): _description_

    Returns:
        list: _description_
    """  # yaml or yml iteratively find all yaml files
    files = glob.glob(f"{path}/**/*.yaml")
    files.extend(glob.glob(f"{path}/**/*.yml"))
    return files


def cli_main():  # noqa: D103
    tags = ["cALzhemiers", "on original arch", "increase head"]
    notes = "new endpoint"
    host = os.uname()[1]
    console.print(f"Running on {host}")
    project = "Reproduce 77"
    if host == "Faraday":
        project = "dev"
    run = wandb.init(
        project=project,
        # project="dev",
        save_code=True,
        sync_tensorboard=True,
        tags=tags,
        notes=notes,
        settings=wandb.Settings(code_dir="."),
    )
    wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
    # find any yaml file
    yaml_files = find_yaml_files(".")
    if len(yaml_files) != 0:
        artifact = wandb.Artifact(name="config", type="yaml")
        artifact.add_dir(
            "configs/",
        )
        console.print("Logging config artifact successfully", yaml_files)
        wandb.log_artifact(artifact)

    print("wandb run id: ", run.id)
    # run.use_artifact("model:latest")
    try:
        # upload config file
        cli = LightningCLI(  # noqa: F841
            model_class=PDBModelModule,
            subclass_mode_model=True,
            subclass_mode_data=True,
            # datamodule_class=PDBModelModule,
            # trainer_defaults=dict(logger=logger)
            save_config_kwargs={"overwrite": True},
            trainer_defaults=dict(
                callbacks=[
                    EarlyStopping(monitor="val_loss", patience=50, mode="min"),
                    LearningRateMonitor(logging_interval="epoch"),
                    ModelCheckpoint(
                        monitor="val_loss",
                        mode="min",
                        save_top_k=1,
                        filename="{epoch}-{val_loss:.2f}",
                        dirpath="/scratch/lab09/GPCRScan/dumps/checkpoints",
                        verbose=True,
                        save_weights_only=False,
                        auto_insert_metric_name=True,
                    ),
                ],
            ),
        )

        console.print(f"Best model path: {cli.trainer.checkpoint_callback.best_model_path}")
        console.print("Testing model")

        cli.trainer.test(
            cli.model,
            ckpt_path=cli.trainer.checkpoint_callback.best_model_path,  # type: ignore
            dataloaders=cli.datamodule.test_dataloader(),
        )

        # add artifact model
        artifact = wandb.Artifact(
            name="model",
            type="model",
            description="final model checkpoint",
        )
        console.print("Logging model artifact")

        artifact.add_file(cli.trainer.checkpoint_callback.best_model_path)  # type: ignore
        # artifact.add_file(save_path)  # type: ignore
        wandb.log_artifact(artifact)
        console.print("Logging config artifact successfully")
        wandb.finish()

    except Exception as e:
        console.print(e)
        pass

    # save_config_kwargs={"overwrite": True}
    # wandb.finish()


if __name__ == "__main__":
    cli_main()

