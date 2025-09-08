#!/usr/bin/env python

"""
Train LTXV models using configuration from YAML files.

This script provides a command-line interface for training LTXV models using
either LoRA fine-tuning or full model fine-tuning. It loads configuration from
a YAML file and passes it to the trainer.

Basic usage:
    train.py --config configs/ltxv_lora_config.yaml
"""

from pathlib import Path

import typer
import yaml
from typing import *
from rich.console import Console
from rich.progress import BarColumn, Group, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from .trainer import CustomTrainerConfig as LtxvTrainerConfig
from .trainer import CustomTrainer as LtxvTrainer

console = Console()
app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True, help="Inference LTXV models using configuration from YAML files.")

@app.command()
def main(config_path: str = typer.Argument(..., help="Path to YAML configuration file")) -> None:
    """Train the model using the provided configuration file."""
    # Load the configuration from the YAML file
    if not Path(config_path).exists():
        typer.echo(f"Error: Configuration file {config_path} does not exist.")
        raise typer.Exit(code=1)

    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)

    # Convert the loaded data to the LtxvTrainerConfig object
    try:
        trainer_config = LtxvTrainerConfig(**config_data)
    except Exception as e:
        typer.echo(f"Error: Invalid configuration data: {e}")
        raise typer.Exit(code=1) from e

    # Initialize the training process
    trainer = LtxvTrainer(trainer_config)
    sample_progress = Progress(
        TextColumn("Sampling validation videos"),
        MofNCompleteColumn(),
        BarColumn(bar_width=40, style="blue"),
        TimeElapsedColumn(),
        TextColumn("ETA:"),
        TimeRemainingColumn(compact=True),
    )
    trainer._sample_videos(sample_progress)


if __name__ == "__main__":
    app()
