"""
Contains various utility functions for PyTorch model training and saving.
"""

import torch
from pathlib import Path
import subprocess
import pyyaml

from typing import Dict


# TODO add pyyaml config docs


def copy_to_drive_async(local_path: Path, drive_path: Path):
    """
    Copies a file asynchronously to a specified location (e.g. Google Drive mount)
    using a subprocess. This is useful for avoiding I/O blocking during training.

    Args:
        local_path (Path): The full path to the local file to be copied.
        drive_path (Path): The target path in the drive where the file will be copied.

    Note:
        This function suppresses all stdout and stderr from the subprocess call.
    """
    subprocess.Popen(
        ["cp", "-f", str(local_path), str(drive_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), (
        "model_name should end with '.pt' or '.pth'"
    )
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def save_checkpoint(
    target_dir: str,
    checkpoint_name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    colab_local_path: str = "/content/ckpts",
    scheduler: torch.optim.lr_scheduler = None,
    scaler: torch.cuda.amp.GradScaler = None,
    test_loss: float = None,
    config_file: str = None,
) -> Dict:
    """
    Saves a complete training checkpoint including model weights, optimiser state,
    scheduler, AMP scaler, loss value, and training configuration file reference.

    Args:
        target_dir (str): Directory where the checkpoint will be saved.
        checkpoint_name (str): Name of the checkpoint file. Must end with '.pt' or '.pth'.
        model (torch.nn.Module): The model whose weights will be saved.
        optimizer (torch.optim.Optimizer): Optimiser instance to save.
        epoch (int): Epoch number to record in the checkpoint.
        colab_drive_path (str, optional): Path to copy the checkpoint asynchronously to Google Drive.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler to save (if used).
        scaler (torch.cuda.amp.GradScaler, optional): AMP gradient scaler state (if using mixed precision).
        test_loss (float, optional): Latest test loss value.
        config_file (str, optional): Path to the training configuration file (YAML/JSON).

    Returns:
        str: Path to the saved checkpoint file.

    Raises:
        AssertionError: If checkpoint_name does not end with '.pt' or '.pth'.
    """

    # input checks
    checkpoint_file_ext = Path(checkpoint_name).suffix
    assert checkpoint_file_ext in ["pt", "pth"], (
        f"[WARNING] Expected checkpoint file extension of .pt or .pth, got: {checkpoint_file_ext}."
    )

    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    checkpoint_filepath = target_dir_path / checkpoint_name

    # populate checkpoint dictionary
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "loss": test_loss if test_loss else None,
        "config_file": config_file if config_file else None,
    }

    # save checkpoint
    torch.save(checkpoint, checkpoint_filepath)

    # if colab drive path provided, perform async copy to maintain UI connection
    if colab_local_path:
        copy_to_drive_async(Path(colab_local_path), checkpoint_filepath)

    return str(checkpoint_filepath)
