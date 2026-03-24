

import os
import sys
import time
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from dataset import get_dataloaders
from model import (
    build_model, freeze_backbone, unfreeze_all,
    get_device, save_checkpoint, NUM_CLASSES, CLASS_NAMES
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)



CONFIG = {
    # --- Paths ---
    "manifest_path": Path("data/manifest.csv"),
    "models_dir":    Path("models"),
    "figures_dir":   Path("notebooks/figures"),


    "batch_size":  32,
   
    "num_workers": 2,
   
    "phase1_epochs": 10,
    "phase1_lr":     1e-3,
   
    "phase2_epochs": 20,
    "phase2_lr":     1e-4,
  

    # --- Regularisation ---
    "weight_decay": 1e-4,
   
    # --- Early stopping ---
    "patience": 5,
  

    # --- Scheduler ---
    "scheduler_factor":   0.5,
    
    "scheduler_patience": 3,
   
}



def train_one_epoch(
    model:      nn.Module,
    loader:     torch.utils.data.DataLoader,
    criterion:  nn.Module,
    optimizer:  torch.optim.Optimizer,
    device:     torch.device,
    epoch:      int,
    total_epochs: int,
) -> tuple[float, float]:
  
    model.train()
  

    running_loss    = 0.0
    correct         = 0
    total           = 0
    n_batches       = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
       
        optimizer.zero_grad()

        # Step 2: Forward pass
        outputs = model(images)
       
        # Step 3: Compute loss
        loss = criterion(outputs, labels)
     

        # Step 4: Backpropagation
        loss.backward()
      
        # Gradient clipping (optional but good practice)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
       
        # Step 5: Update weights
        optimizer.step()
       

        # Track metrics
        running_loss += loss.item() * images.size(0)
     

        _, predicted = outputs.max(dim=1)
      

        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == n_batches:
            log.info(
                f"  Epoch {epoch}/{total_epochs}  "
                f"Batch {batch_idx+1}/{n_batches}  "
                f"Loss: {running_loss/total:.4f}  "
                f"Acc: {100.*correct/total:.2f}%"
            )

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy



def evaluate(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float]:
    
    model.eval()

    running_loss = 0.0
    correct      = 0
    total        = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted  = outputs.max(dim=1)
            correct       += predicted.eq(labels).sum().item()
            total         += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


class EarlyStopping:
   
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
     
        self.best_loss  = float('inf')
        self.counter    = 0
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
      
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            return False  # improved — continue training
        else:
            self.counter += 1
            log.info(f"  Early stopping: {self.counter}/{self.patience} epochs without improvement")
            if self.counter >= self.patience:
                self.should_stop = True
                return True  # stop training
            return False


def plot_training_curves(history: dict, save_path: Path) -> None:
  
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    axes[0].plot(epochs, history["train_loss"], 'b-o', markersize=3, label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   'r-o', markersize=3, label="Val Loss")
    if "phase2_start" in history:
        axes[0].axvline(x=history["phase2_start"], color='green', linestyle='--',
                        label=f"Phase 2 starts (epoch {history['phase2_start']})")
    axes[0].set_title("Loss per Epoch", fontweight='bold')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, [a*100 for a in history["train_acc"]], 'b-o', markersize=3, label="Train Acc")
    axes[1].plot(epochs, [a*100 for a in history["val_acc"]],   'r-o', markersize=3, label="Val Acc")
    if "phase2_start" in history:
        axes[1].axvline(x=history["phase2_start"], color='green', linestyle='--',
                        label=f"Phase 2 starts")
    axes[1].set_title("Accuracy per Epoch", fontweight='bold')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 105)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("PathoLens — EfficientNet-B0 Training Curves", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Training curves saved → {save_path}")


def train():
    log.info("=" * 60)
    log.info("  PathoLens — Training EfficientNet-B0")
    log.info("=" * 60)

    # Setup
    device  = get_device()
    CONFIG["models_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["figures_dir"].mkdir(parents=True, exist_ok=True)

    # ── DATA ──────────────────────────────────────────────────────────────────
    log.info("Loading data...")
    dataloaders, datasets = get_dataloaders(
        manifest_path=CONFIG["manifest_path"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
    )

    # ── MODEL ─────────────────────────────────────────────────────────────────
    log.info("Building model...")
    model = build_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    # History tracking
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }
    best_val_loss = float('inf')
    best_model_path = CONFIG["models_dir"] / "best_model.pth"

    log.info("")
    log.info("━" * 60)
    log.info("  PHASE 1 — Training classification head only")
    log.info(f"  Epochs: {CONFIG['phase1_epochs']}  |  LR: {CONFIG['phase1_lr']}")
    log.info("━" * 60)

  
    optimizer_p1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["phase1_lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    scheduler_p1 = ReduceLROnPlateau(
        optimizer_p1,
        mode='min',        # reduce LR when loss stops decreasing
        factor=CONFIG["scheduler_factor"],
        patience=CONFIG["scheduler_patience"],
    )

    early_stop_p1 = EarlyStopping(patience=CONFIG["patience"])

    for epoch in range(1, CONFIG["phase1_epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer_p1, device, epoch, CONFIG["phase1_epochs"]
        )
        val_loss, val_acc = evaluate(model, dataloaders["val"], criterion, device)

        elapsed = time.time() - t0
        log.info(
            f"Epoch {epoch:>2}/{CONFIG['phase1_epochs']}  "
            f"| train_loss={train_loss:.4f}  train_acc={train_acc*100:.2f}%"
            f"| val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%"
            f"| {elapsed:.0f}s"
        )

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer_p1, epoch, val_loss, val_acc, best_model_path)

        # LR scheduler step
        scheduler_p1.step(val_loss)

        # Early stopping check
        if early_stop_p1(val_loss):
            log.info(f"Early stopping triggered at epoch {epoch}")
            break

    log.info("")
    log.info("━" * 60)
    log.info("  PHASE 2 — Fine-tuning all layers")
    log.info(f"  Epochs: {CONFIG['phase2_epochs']}  |  LR: {CONFIG['phase2_lr']}")
    log.info("━" * 60)

    history["phase2_start"] = len(history["train_loss"]) + 1

    # Unfreeze all backbone layers
    unfreeze_all(model)

    # New optimizer covering ALL parameters at a lower LR
    optimizer_p2 = optim.Adam(
        model.parameters(),
        lr=CONFIG["phase2_lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    scheduler_p2 = ReduceLROnPlateau(
        optimizer_p2,
        mode='min',
        factor=CONFIG["scheduler_factor"],
        patience=CONFIG["scheduler_patience"],
    )

    early_stop_p2 = EarlyStopping(patience=CONFIG["patience"])

    for epoch in range(1, CONFIG["phase2_epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer_p2, device,
            epoch, CONFIG["phase2_epochs"]
        )
        val_loss, val_acc = evaluate(model, dataloaders["val"], criterion, device)

        elapsed = time.time() - t0
        log.info(
            f"Epoch {epoch:>2}/{CONFIG['phase2_epochs']}  "
            f"| train_loss={train_loss:.4f}  train_acc={train_acc*100:.2f}%  "
            f"| val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%  "
            f"| {elapsed:.0f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Save best model (continues from Phase 1 — can still improve)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer_p2, epoch, val_loss, val_acc, best_model_path)

        scheduler_p2.step(val_loss)

        if early_stop_p2(val_loss):
            log.info(f"Early stopping triggered at epoch {epoch}")
            break

    # ── WRAP UP ───────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("  Training complete.")
    log.info(f"  Best val loss    : {best_val_loss:.4f}")
    log.info(f"  Best model saved : {best_model_path}")
    log.info("=" * 60)

    # Save training history as JSON
    history_path = CONFIG["models_dir"] / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info(f"Training history saved → {history_path}")

    # Plot training curves
    plot_training_curves(history, CONFIG["figures_dir"] / "08_training_curves.png")

    return model, history



if __name__ == "__main__":
    train()
