import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
)
from PIL import Image
import torchvision.transforms as transforms

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from src.dataset import get_dataloaders, get_val_transforms, HistoDataset
from src.model import load_model_for_inference, get_device, NUM_CLASSES, CLASS_NAMES


# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

CONFIG = {
    "manifest_path":   Path("data/manifest.csv"),
    "model_path":      Path("models/best_model.pth"),
    "figures_dir":     Path("notebooks/figures"),
    "metrics_path":    Path("models/evaluation_metrics.json"),
    "batch_size":      32,
    "num_workers":     2,

    # Grad-CAM: how many sample images to visualise per class
    "gradcam_samples_per_class": 2,

    # Confidence threshold below which a prediction is flagged for review
    # Used by the report generator — documented here for consistency
    "low_confidence_threshold": 0.80,
}

# Short labels for plots
SHORT_LABELS = {
    "Colon Adenocarcinoma":         "Colon ACA",
    "Colon Benign":                 "Colon Benign",
    "Lung Adenocarcinoma":          "Lung ACA",
    "Lung Benign":                  "Lung Benign",
    "Lung Squamous Cell Carcinoma": "Lung SCC",
}

MALIGNANT = {0, 2, 4}   # class indices that are malignant
COLORS    = ["#C0392B", "#27AE60", "#E74C3C", "#2ECC71", "#8E44AD"]


# =============================================================================
# STEP 1 — COLLECT ALL PREDICTIONS ON THE TEST SET
# =============================================================================

def get_predictions(model, loader, device):
   
    model.eval()

    all_true  = []
    all_pred  = []
    all_probs = []

    log.info("Running inference on test set...")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            probs  = F.softmax(logits, dim=1)
           

            preds  = probs.argmax(dim=1)

            all_true.extend(labels.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            if (batch_idx + 1) % 20 == 0:
                log.info(f"  Processed {(batch_idx+1)*loader.batch_size}/{len(loader.dataset)} images")

    return (
        np.array(all_true),
        np.array(all_pred),
        np.array(all_probs),   
    )


# =============================================================================
# STEP 2 — COMPUTE CLINICAL METRICS
# =============================================================================

def compute_clinical_metrics(y_true, y_pred, y_probs):
   
    metrics = {}
    n_classes = len(CLASS_NAMES)

    # ── OVERALL ACCURACY ─────────────────────────────────────────────────────
    overall_acc = (y_true == y_pred).mean()
    metrics["overall_accuracy"] = float(overall_acc)
    log.info(f"Overall Accuracy : {overall_acc*100:.2f}%")

   
    cm = confusion_matrix(y_true, y_pred)
  

    per_class = {}

    for i, class_name in enumerate(CLASS_NAMES):
        

        tp = cm[i, i]
        

        fn = cm[i, :].sum() - tp
        

        fp = cm[:, i].sum() - tp
        

        tn = cm.sum() - tp - fn - fp
        

        sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv          = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv          = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        

        # AUC-ROC for this class (One-vs-Rest)
        binary_true = (y_true == i).astype(int)
        class_probs = y_probs[:, i]
        try:
            fpr, tpr, _ = roc_curve(binary_true, class_probs)
            auc_score   = auc(fpr, tpr)
        except Exception:
            auc_score = 0.0

        per_class[class_name] = {
            "sensitivity":  round(float(sensitivity), 4),
            "specificity":  round(float(specificity), 4),
            "ppv":          round(float(ppv), 4),
            "npv":          round(float(npv), 4),
            "auc_roc":      round(float(auc_score), 4),
            "tp": int(tp), "fn": int(fn),
            "fp": int(fp), "tn": int(tn),
            "malignant":    i in MALIGNANT,
        }

        log.info(
            f"  {SHORT_LABELS[class_name]:<14} "
            f"Sensitivity={sensitivity*100:.1f}%  "
            f"Specificity={specificity*100:.1f}%  "
            f"AUC={auc_score:.4f}"
        )

    metrics["per_class"]         = per_class
    metrics["confusion_matrix"]  = cm.tolist()
    metrics["class_names"]       = CLASS_NAMES
    metrics["short_labels"]      = SHORT_LABELS

    # ── MACRO AUC-ROC ─────────────────────────────────────────────────────────
    
    try:
        macro_auc = roc_auc_score(
            y_true, y_probs,
            multi_class='ovr',   # One-vs-Rest
            average='macro'
        )
        metrics["macro_auc_roc"] = round(float(macro_auc), 4)
        log.info(f"Macro AUC-ROC    : {macro_auc:.4f}")
    except Exception as e:
        log.warning(f"Could not compute macro AUC: {e}")
        metrics["macro_auc_roc"] = None

    # ── SKLEARN CLASSIFICATION REPORT ─────────────────────────────────────────
    report = classification_report(
        y_true, y_pred,
        target_names=[SHORT_LABELS[c] for c in CLASS_NAMES],
        output_dict=True,
    )
    metrics["classification_report"] = report

    return metrics, cm


# =============================================================================
# STEP 3 — CONFUSION MATRIX PLOT
# =============================================================================

def plot_confusion_matrix(cm, save_path):
   
    short = [SHORT_LABELS[c] for c in CLASS_NAMES]

    # Normalised (row percentages) for colour
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        "PathoLens — Confusion Matrix\nEfficientNet-B0 on LC25000 Test Set",
        fontsize=14, fontweight='bold'
    )

    for ax, data, title, fmt in [
        (axes[0], cm,      "Raw Counts",          "d"),
        (axes[1], cm_norm, "Normalised (row %)",  ".2%"),
    ]:
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=short,
            yticklabels=short,
            ax=ax,
            linewidths=0.5,
            linecolor='white',
            cbar_kws={'shrink': 0.8},
        )
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Predicted Class", fontsize=11)
        ax.set_ylabel("True Class", fontsize=11)
        ax.tick_params(axis='x', rotation=30)
        ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Confusion matrix saved → {save_path}")


# =============================================================================
# STEP 4 — ROC CURVES
# =============================================================================

def plot_roc_curves(y_true, y_probs, save_path):
   
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    fig.suptitle(
        "PathoLens — ROC Curves per Class\nEfficientNet-B0 on LC25000 Test Set",
        fontsize=14, fontweight='bold'
    )

    auc_scores = []

    for i, (class_name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
        ax = axes[i]

        binary_true = (y_true == i).astype(int)
        class_probs = y_probs[:, i]

        fpr, tpr, thresholds = roc_curve(binary_true, class_probs)
        auc_score = auc(fpr, tpr)
        auc_scores.append(auc_score)

        # ROC curve
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f"AUC = {auc_score:.4f}")

        # Random classifier baseline
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label="Random")

        # Shade the area under the curve
        ax.fill_between(fpr, tpr, alpha=0.1, color=color)

        
        idx = np.argmin(np.abs(thresholds - 0.5))
        ax.scatter(fpr[idx], tpr[idx], color=color, s=80, zorder=5,
                   label=f"t=0.5: TPR={tpr[idx]:.3f}")

        is_mal = i in MALIGNANT
        title_suffix = " 🔴 MALIGNANT" if is_mal else " 🟢 Benign"
        ax.set_title(f"{SHORT_LABELS[class_name]}{title_suffix}",
                     fontsize=10, fontweight='bold')
        ax.set_xlabel("False Positive Rate (1 - Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity)")
        ax.legend(fontsize=9)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.grid(True, alpha=0.3)

    # Summary panel (6th subplot)
    ax_summary = axes[5]
    ax_summary.axis('off')

    summary_text = "AUC-ROC Summary\n" + "─" * 22 + "\n"
    for name, score in zip(CLASS_NAMES, auc_scores):
        bar = "█" * int(score * 20)
        summary_text += f"{SHORT_LABELS[name]:<14} {score:.4f}\n"
    summary_text += "\n" + "─" * 22
    summary_text += f"\nMacro AUC: {np.mean(auc_scores):.4f}"

    ax_summary.text(
        0.1, 0.5, summary_text,
        transform=ax_summary.transAxes,
        fontsize=11, verticalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#EBF5FB', alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"ROC curves saved → {save_path}")


# =============================================================================
# STEP 5 — CLINICAL METRICS SUMMARY PLOT
# =============================================================================

def plot_clinical_metrics(metrics, save_path):
   
    per_class = metrics["per_class"]
    short     = [SHORT_LABELS[c] for c in CLASS_NAMES]

    sensitivity  = [per_class[c]["sensitivity"]  for c in CLASS_NAMES]
    specificity  = [per_class[c]["specificity"]  for c in CLASS_NAMES]
    auc_scores   = [per_class[c]["auc_roc"]      for c in CLASS_NAMES]

    x     = np.arange(len(CLASS_NAMES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - width, sensitivity, width, label="Sensitivity",
                   color="#2980B9", alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x,          specificity, width, label="Specificity",
                   color="#27AE60", alpha=0.85, edgecolor='white')
    bars3 = ax.bar(x + width,  auc_scores,  width, label="AUC-ROC",
                   color="#8E44AD", alpha=0.85, edgecolor='white')

    # Value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.002,
                f"{h:.3f}",
                ha='center', va='bottom', fontsize=8, fontweight='bold'
            )

    # Malignant class indicators
    for i, class_name in enumerate(CLASS_NAMES):
        if i in MALIGNANT:
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.05, color='red')

    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=20, ha='right')
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_title(
        "PathoLens — Clinical Evaluation Metrics per Class\n"
        "(Red shading = malignant classes)",
        fontsize=13, fontweight='bold'
    )
    ax.legend(loc='lower right')
    ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.text(4.6, 0.952, "0.95", fontsize=8, color='gray')
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Clinical metrics plot saved → {save_path}")


# =============================================================================
# STEP 6 — GRAD-CAM VISUALISATIONS
# =============================================================================


class GradCAM:


    def __init__(self, model, target_layer_name="blocks"):
        self.model    = model
        self.gradient = None
        self.activation = None

       
        target_layer = None
        for name, module in model.named_modules():
            if target_layer_name in name:
                target_layer = module

        if target_layer is None:
            raise ValueError(f"Layer '{target_layer_name}' not found in model.")

        
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activation = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0].detach()

    def generate(self, image_tensor, class_idx):
      
        self.model.eval()
        image_tensor = image_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(image_tensor)

        # Backward pass for the target class only
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()

        
        weights = self.gradient.mean(dim=(2, 3)).squeeze()

       
        cam = torch.zeros(self.activation.shape[2:], device=self.activation.device)
        for i, w in enumerate(weights):
            cam += w * self.activation[0, i]

        
        cam = F.relu(cam)

        # Normalise to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        # Upsample to input image size
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            mode='bilinear',
            align_corners=False,
        ).squeeze().cpu().numpy()

        return cam

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def overlay_heatmap(image_np, cam, alpha=0.45):
    
    # Convert CAM to a colour heatmap (jet colormap)
    cmap    = plt.cm.jet
    heatmap = cmap(cam)[:, :, :3]                 # drop alpha channel
    heatmap = (heatmap * 255).astype(np.uint8)

    # Blend with original image
    image_float   = image_np.astype(float)
    heatmap_float = heatmap.astype(float)
    overlay       = (1 - alpha) * image_float + alpha * heatmap_float
    return np.clip(overlay, 0, 255).astype(np.uint8)


def plot_gradcam(model, manifest_path, device, save_path, n_per_class=2):
   
    log.info("Generating Grad-CAM visualisations...")

    df      = pd.read_csv(manifest_path)
    gradcam = GradCAM(model)

    
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    val_transforms = get_val_transforms()

    n_classes = len(CLASS_NAMES)
    n_cols    = n_per_class * 3   
    fig, axes = plt.subplots(
        n_classes, n_cols,
        figsize=(n_cols * 2.8, n_classes * 3.2)
    )

    fig.suptitle(
        "PathoLens — Grad-CAM: What Does The Model Look At?\n"
        "Each row: Original | Heatmap | Overlay",
        fontsize=13, fontweight='bold', y=1.01
    )

    for row_idx, class_name in enumerate(CLASS_NAMES):
        # Sample images for this class from the test set
        class_df = df[(df['label'] == class_name) & (df['split'] == 'test')]
        samples  = class_df.sample(
            n=min(n_per_class, len(class_df)), random_state=42
        )

        for sample_idx, (_, row) in enumerate(samples.iterrows()):
            col_base = sample_idx * 3

            # Load image
            pil_img   = Image.open(row['image_path']).convert('RGB')
            pil_resized = pil_img.resize((224, 224))
            img_np    = np.array(pil_resized)

            # Prepare tensor for model
            tensor = val_transforms(pil_img).unsqueeze(0).to(device)

            # Generate Grad-CAM for the true class
            true_class = int(row['label_index'])
            cam = gradcam.generate(tensor, true_class)

            overlay = overlay_heatmap(img_np, cam)

            # Plot original
            axes[row_idx, col_base].imshow(img_np)
            axes[row_idx, col_base].axis('off')
            if row_idx == 0:
                axes[row_idx, col_base].set_title(f"Sample {sample_idx+1}\nOriginal",
                                                   fontsize=8)

            # Plot heatmap
            axes[row_idx, col_base+1].imshow(cam, cmap='jet', vmin=0, vmax=1)
            axes[row_idx, col_base+1].axis('off')
            if row_idx == 0:
                axes[row_idx, col_base+1].set_title("Activation\nHeatmap", fontsize=8)

            # Plot overlay
            axes[row_idx, col_base+2].imshow(overlay)
            axes[row_idx, col_base+2].axis('off')
            if row_idx == 0:
                axes[row_idx, col_base+2].set_title("Overlay", fontsize=8)

        # Class label on the left
        is_mal = CLASS_NAMES.index(class_name) in MALIGNANT
        label  = f"{'🔴' if is_mal else '🟢'} {SHORT_LABELS[class_name]}"
        axes[row_idx, 0].set_ylabel(label, fontsize=10, fontweight='bold',
                                     rotation=0, labelpad=120, va='center')

    gradcam.remove_hooks()
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    log.info(f"Grad-CAM visualisations saved → {save_path}")


# =============================================================================
# STEP 7 — PRINT CLINICAL SUMMARY TABLE
# =============================================================================

def print_clinical_summary(metrics):
  
    per_class = metrics["per_class"]

    print("\n" + "=" * 78)
    print("  PATHOLENS — CLINICAL EVALUATION SUMMARY")
    print("  EfficientNet-B0 | LC25000 Test Set | n=2,499")
    print("=" * 78)
    print(f"  {'Class':<30} {'Sens':>6} {'Spec':>6} {'AUC':>6} {'PPV':>6} {'Type':<10}")
    print("  " + "-" * 74)

    for class_name in CLASS_NAMES:
        m    = per_class[class_name]
        typ  = "MALIGNANT" if m["malignant"] else "Benign"
        flag = "⚠" if m["malignant"] else " "
        print(
            f"  {flag} {SHORT_LABELS[class_name]:<28} "
            f"{m['sensitivity']:>5.1%} "
            f"{m['specificity']:>5.1%} "
            f"{m['auc_roc']:>5.4f} "
            f"{m['ppv']:>5.1%} "
            f"{typ}"
        )

    print("  " + "-" * 74)
    print(f"  Overall Accuracy : {metrics['overall_accuracy']*100:.2f}%")
    print(f"  Macro AUC-ROC    : {metrics.get('macro_auc_roc', 'N/A')}")
    print("=" * 78)

    print("\n  CLINICAL INTERPRETATION")
    print("  " + "-" * 74)
    for class_name in CLASS_NAMES:
        m = per_class[class_name]
        if m["malignant"]:
            fn_count = m["fn"]
            print(
                f"  {SHORT_LABELS[class_name]}: "
                f"{fn_count} false negatives out of {m['tp']+m['fn']} true cases. "
                f"Sensitivity = {m['sensitivity']:.1%}"
            )
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    log.info("=" * 60)
    log.info("  PathoLens — Clinical Evaluation Suite")
    log.info("=" * 60)

    CONFIG["figures_dir"].mkdir(parents=True, exist_ok=True)

    # ── DEVICE AND MODEL ──────────────────────────────────────────────────────
    device = get_device()
    model  = load_model_for_inference(CONFIG["model_path"], device)

    # ── DATA ──────────────────────────────────────────────────────────────────
    dataloaders, _ = get_dataloaders(
        CONFIG["manifest_path"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
    )
    test_loader = dataloaders["test"]

    # ── PREDICTIONS ───────────────────────────────────────────────────────────
    y_true, y_pred, y_probs = get_predictions(model, test_loader, device)

    # ── METRICS ───────────────────────────────────────────────────────────────
    log.info("Computing clinical metrics...")
    metrics, cm = compute_clinical_metrics(y_true, y_pred, y_probs)

    # ── PLOTS ─────────────────────────────────────────────────────────────────
    log.info("Generating evaluation plots...")

    plot_confusion_matrix(
        cm,
        CONFIG["figures_dir"] / "09_confusion_matrix.png"
    )

    plot_roc_curves(
        y_true, y_probs,
        CONFIG["figures_dir"] / "10_roc_curves.png"
    )

    plot_clinical_metrics(
        metrics,
        CONFIG["figures_dir"] / "11_clinical_metrics.png"
    )

    plot_gradcam(
        model,
        CONFIG["manifest_path"],
        device,
        CONFIG["figures_dir"] / "12_gradcam.png",
        n_per_class=CONFIG["gradcam_samples_per_class"],
    )

    # ── PRINT SUMMARY ─────────────────────────────────────────────────────────
    print_clinical_summary(metrics)

   
    CONFIG["metrics_path"].parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG["metrics_path"], "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metrics saved → {CONFIG['metrics_path']}")

    log.info("=" * 60)
    log.info("  Evaluation complete.")
    log.info(f"  Figures saved to: {CONFIG['figures_dir']}")
    log.info("=" * 60)

    return metrics


if __name__ == "__main__":
    main()
