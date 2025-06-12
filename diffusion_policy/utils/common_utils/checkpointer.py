import heapq
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class Checkpointer:
    def __init__(self, save_dir: Path, max_checkpoints: int = 5):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.best_checkpoints = []
        self.latest_checkpoint = None

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_loss: float,
    ):
        checkpoint = {
            "model_state_dict": model.state_dict()
            if not hasattr(model, "_orig_mod")
            else model._orig_mod.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        }

        checkpoint_path = (
            self.save_dir / f"checkpoint_epoch_{epoch}_val_loss_{val_loss:.5f}.pth"
        )
        torch.save(checkpoint, checkpoint_path)

        # Manage best checkpoints
        if len(self.best_checkpoints) < self.max_checkpoints:
            heapq.heappush(self.best_checkpoints, (-val_loss, checkpoint_path))
        else:
            heapq.heappushpop(self.best_checkpoints, (-val_loss, checkpoint_path))

        # Remove checkpoints that are not in the best k
        all_checkpoints = list(self.save_dir.glob("checkpoint_*.pth"))
        best_checkpoint_paths = {path for _, path in self.best_checkpoints}
        for checkpoint_path in all_checkpoints:
            if checkpoint_path not in best_checkpoint_paths:
                checkpoint_path.unlink()

    def save_last(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_loss: float,
    ):
        checkpoint = {
            "model_state_dict": model.state_dict()
            if not hasattr(model, "_orig_mod")
            else model._orig_mod.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        }

        checkpoint_path = self.save_dir / f"ckpt_last_val_loss_{val_loss:.5f}.pth"
        torch.save(checkpoint, checkpoint_path)

    def load_best(
        self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None
    ):
        if not self.best_checkpoints:
            return None
        _, best_checkpoint_path = min(self.best_checkpoints)
        # _, best_checkpoint_path = max(self.best_checkpoints)
        checkpoint = torch.load(best_checkpoint_path)
        if hasattr(model, "_orig_mod"):
            model._orig_mod.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint

    def load_latest(
        self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None
    ):
        if self.latest_checkpoint is None:
            return None
        checkpoint = torch.load(self.latest_checkpoint)
        if hasattr(model, "_orig_mod"):
            model._orig_mod.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint

    def update_checkpoint_from_dir(self, experiment_dir: str) -> None:
        """
        Updates the best_checkpoints list with checkpoint files found in the experiment directory's
        'checkpoints' subdirectory.

        Args:
            experiment_dir (str): The path to the experiment directory.
        """
        if not os.path.exists(experiment_dir):
            print(f"No checkpoints directory found in {experiment_dir}")
            return

        # List all files in the checkpoints directory
        checkpoint_files = []
        for ckpt_name in os.listdir(experiment_dir):
            ckpt_path = os.path.join(experiment_dir, ckpt_name)
            if os.path.isfile(ckpt_path) and "last" not in ckpt_name:
                val_loss = -float(ckpt_name.split("_")[-1].replace(".pth", ""))
                checkpoint_files.append((val_loss, ckpt_path))
            elif "last" in ckpt_name:
                self.latest_checkpoint = ckpt_path

        # Update the best_checkpoints list
        self.best_checkpoints.extend(checkpoint_files)
        print(
            f"Updated best checkpoints: {self.best_checkpoints} and latest checkpoint: {self.latest_checkpoint}"
        )

    @staticmethod
    def load_checkpoint(
        policy: Any,
        checkpointer: Any,
        checkpoint_type: str,
        optim: Optional[Any] = None,
    ) -> Any:
        """
        Loads a checkpoint into the policy model if one exists, optionally using an optimizer.

        Args:
            checkpointer: The checkpointer object responsible for loading checkpoints.
            checkpoint_type: A string indicating the type of checkpoint to load ("best" or "latest").
            policy: The policy model into which the checkpoint will be loaded.
            optim: Optional; the optimizer to be used in conjunction with the policy model.

        Returns:
            The policy model with the loaded checkpoint, potentially recompiled.
        """
        assert checkpoint_type in ["best", "latest"]
        if checkpoint_type == "best":
            loaded_checkpoint = checkpointer.load_best(policy, optim)
        else:
            loaded_checkpoint = checkpointer.load_latest(policy, optim)

        if loaded_checkpoint is not None:
            epoch = loaded_checkpoint["epoch"]
            val_loss = loaded_checkpoint["val_loss"]
            print(
                f"Loaded {checkpoint_type} checkpoint from epoch {epoch} with validation loss {val_loss:.5f}"
            )
        else:
            print("No checkpoint found. Using the current model state.")

        # Recompile the model if necessary
        if hasattr(policy, "_orig_mod"):
            policy = torch.compile(policy._orig_mod)
        else:
            policy = torch.compile(policy)
        return policy, loaded_checkpoint
