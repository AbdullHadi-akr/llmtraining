"""Command-line entry points for the workflow package."""

from .train import train_from_config

__all__ = ["train_from_config"]
