

import torch
import torch.nn as nn
import timm
from pathlib import Path


NUM_CLASSES = 5

CLASS_NAMES = [
    "Colon Adenocarcinoma",          
    "Colon Benign",                  
    "Lung Adenocarcinoma",          
    "Lung Benign",                   
    "Lung Squamous Cell Carcinoma",  
]

MALIGNANT_INDICES = {0, 2, 4}   
BENIGN_INDICES    = {1, 3}     


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=pretrained,
       
    )

    in_features = model.classifier.in_features


    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
    
        nn.Linear(in_features, num_classes),
     
    )


    freeze_backbone(model)

    total_params   = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model: EfficientNet-B0")
    print(f"  Total parameters     : {total_params:,}")
    print(f"  Trainable parameters : {trainable_params:,}  (Phase 1 — head only)")
    print(f"  Frozen parameters    : {total_params - trainable_params:,}")
    print(f"  Output classes       : {num_classes}")

    return model


def freeze_backbone(model: nn.Module) -> None:
  

    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Backbone frozen. Trainable parameters: {trainable:,} (head only)")


def unfreeze_all(model: nn.Module) -> None:
    
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"All layers unfrozen. Trainable parameters: {trainable:,}")


def get_device() -> torch.device:
   
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("Device: CPU  ⚠  Training will be slow. Use Colab for GPU.")
    return device


def save_checkpoint(
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    val_loss:  float,
    val_acc:   float,
    path:      str | Path,
) -> None:
   
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":            epoch,
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "val_loss":         val_loss,
        "val_acc":          val_acc,
        "num_classes":      NUM_CLASSES,
        "class_names":      CLASS_NAMES,
        "architecture":     "efficientnet_b0",
    }, path)
    print(f"Checkpoint saved → {path}  (epoch {epoch}, val_acc={val_acc:.4f})")


def load_model_for_inference(path: str | Path, device: torch.device) -> nn.Module:
    
    checkpoint = torch.load(path, map_location=device)
   

    model = build_model(
        num_classes=checkpoint.get("num_classes", NUM_CLASSES),
        pretrained=False,
    
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
   

    print(f"Model loaded from {path}")
    print(f"  Trained for {checkpoint.get('epoch', '?')} epochs")
    print(f"  Best val accuracy: {checkpoint.get('val_acc', '?'):.4f}")

    return model



if __name__ == "__main__":
    print("=" * 55)
    print("  PathoLens — Model Sanity Check")
    print("=" * 55)
    print()

    device = get_device()
    model  = build_model(num_classes=NUM_CLASSES, pretrained=False)
  
    model.to(device)

   
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    print()
    print(f"Input shape  : {dummy_input.shape}")

    model.eval()
    with torch.no_grad():
       
        output = model(dummy_input)

    print(f"Output shape : {output.shape}")
   
    probs = torch.softmax(output, dim=1)
    print(f"Probs shape  : {probs.shape}")
    print(f"Probs sum    : {probs.sum(dim=1).tolist()}")
  

    print()
    print("Phase 2 — unfreeze all:")
    unfreeze_all(model)

    print()
    print("All checks passed. model.py is working correctly.")
