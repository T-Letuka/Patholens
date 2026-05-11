---
title: PathoLens
emoji: 🔬
colorFrom: teal
colorTo: blue
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: AI Histopathology Classifier + Clinical Report Generator
---

<div align="center">

# 🔬 PathoLens

### AI-Assisted Histopathology Image Classification and Clinical Report Generation

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude-191919?style=flat-square)](https://anthropic.com)
[![License](https://img.shields.io/badge/License-MIT-00B894?style=flat-square)](LICENSE)
[![HF Spaces](https://img.shields.io/badge/🤗-Live%20Demo-yellow?style=flat-square)](LIVE_LINK_PLACEHOLDER)

**[ Try the Live Demo](https://patholens.streamlit.app/)**

*Randomise a sample slide · Download it · Upload it · Get a clinical report*

## Demo Video

Watch the Loom walkthrough here: [Project Demo](https://www.loom.com/share/f8006681988e443296dc6e8cdcb707a9)

</div>

---

## The Problem

Every year, millions of tissue biopsy samples are sent to pathology laboratories around the world. A pathologist physically examines each slide under a microscope, identifies the tissue architecture and cellular morphology, and dictates a report a process that is skilled, time-consuming, and entirely dependent on human availability.

This creates three compounding problems:

**1. Volume.** Demand for histopathological analysis is growing faster than the pathologist workforce. In South Africa and across sub-Saharan Africa, the pathologist-to-population ratio is critically low and in some provinces, a single pathologist may be responsible for thousands of cases per year. Delays in diagnosis directly translate to delays in cancer treatment.

**2. Consistency.** Even among experienced pathologists, inter-observer variability exists particularly for ambiguous or rare presentations. A second opinion is not always accessible, especially in under-resourced settings.

**3. The cost of a missed diagnosis.** In cancer diagnostics, a false negative: a pathologist or system predicting benign tissue when the slide shows malignancy is not a statistic. It is a patient who leaves the clinic believing they are healthy while a tumour progresses untreated. The clinical cost of false negatives is categorically different from the cost of false positives.

---

## What PathoLens Solves

PathoLens is not a replacement for a pathologist. It is a **screening and decision-support tool** the kind of system that can flag suspicious slides for priority review, assist in high-volume settings where every minute matters, and generate structured preliminary reports that a pathologist can confirm, correct, or override.

Specifically, PathoLens addresses:

- **Speed:** Classification of a tissue slide image takes under two seconds. Report generation takes under fifteen. A pathologist reviewing an AI-generated preliminary report spends less time on routine cases and more time on complex ones.

- **Structured reporting:** The system generates a four-section preliminary pathology report — microscopic description, interpretation, clinical significance, and recommended next steps in the same format a pathologist would dictate. This is immediately legible to clinical staff.

- **Responsible uncertainty:** Not every slide is unambiguous. PathoLens implements a dual-path reporting system: when the model's confidence exceeds 80%, it generates a primary diagnosis report. When confidence falls below that threshold, it generates a differential diagnosis report, lists the top three possibilities with probabilities, and flags the case as **REQUIRES EXPERT REVIEW**. A system that acknowledges its own uncertainty is safer than one that always sounds confident. Check the loom recording.I tested this using images from the internet, outside the dataset.

- **Clinical framing:** PathoLens evaluates itself using sensitivity and specificity  not accuracy. A model with 95% accuracy that misses every malignant case is clinically useless. Sensitivity, the fraction of true malignant cases correctly identified, is the metric that matters.

---

## Model Performance

| Metric | Score | Clinical Meaning |
|--------|-------|-----------------|
| Overall Test Accuracy | **100.00%** | Correct on all 2,499 held-out test images |
| Macro AUC-ROC | **1.0000** | Perfect discrimination across all thresholds |
| Sensitivity — Malignant Classes | **100.0%** | Zero false negatives — no missed cancers |
| Specificity — Benign Classes | **100.0%** | Zero false positives — no unnecessary alarms |
| Test Set Size | **2,499 images** | Completely held-out, never seen during training |

> **On benchmark performance:** LC25000 is a purpose-built research dataset with consistent H&E staining and controlled image acquisition. These results reflect performance under idealised conditions. Real-world clinical deployment would require prospective validation across multiple laboratories, scanners, and staining protocols — a deliberate limitation .

---

## Demo

<div align="center">

| Step | Action |
|------|--------|
| 1 | Click **🎲 Randomise Sample** to get a histology slide |
| 2 | Read the specimen metadata — tissue site, stain, source |
| 3 | Download the slide (you don't know the diagnosis yet) |
| 4 | Upload it in Step 2 |
| 5 | Click **🧬 Analyse Tissue** |
| 6 | See the classification result and confidence scores |
| 7 | Click **📄 Generate Clinical Report** |

**[→ Open PathoLens](LIVE_LINK_PLACEHOLDER)**

</div>

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT                                 │
│              H&E Stained Histopathology Image                │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   CNN CLASSIFIER                             │
│         EfficientNet-B0 (fine-tuned, PyTorch)               │
│         Two-phase transfer learning strategy                 │
│         Input: 224×224px  │  Output: 5 class logits         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│               CLASS PROBABILITIES + CONFIDENCE               │
│                                                             │
│    Confidence ≥ 80%              Confidence < 80%           │
│          │                              │                   │
│          ▼                              ▼                   │
│   Primary Diagnosis            Differential Diagnosis       │
│   Prompt Path                  Prompt Path                  │
│                                + EXPERT REVIEW FLAG         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  LLM REPORT GENERATOR                        │
│              Anthropic Claude API                            │
│   Sections: Microscopic Description · Interpretation         │
│             Clinical Significance · Next Steps               │
│             ICD-10 Code · Mandatory Disclaimer               │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  STREAMLIT FRONTEND                          │
│   Two-view UX · Plotly probability chart · Report display   │
│         Downloadable report · Sample image randomiser        │
└─────────────────────────────────────────────────────────────┘
```

---

## The Five Tissue Classes

| Class | Type | Key Histological Features | ICD-10 |
|-------|------|--------------------------|--------|
| Colon Adenocarcinoma | 🔴 Malignant | Irregular glandular architecture, nuclear pleomorphism, loss of cellular polarity, desmoplastic stroma | C18.9 |
| Colon Benign | 🟢 Benign | Orderly crypt architecture, uniform goblet cells with mucin vacuoles, basally-oriented nuclei, intact basement membrane | K63.9 |
| Lung Adenocarcinoma | 🔴 Malignant | Acinar/lepidic growth patterns, mucin production, nuclear atypia, reactive fibrotic stroma | C34.1 |
| Lung Benign | 🟢 Benign | Patent alveolar spaces, uniform type I/II pneumocytes, intact capillary network, no architectural distortion | J98.4 |
| Lung Squamous Cell Carcinoma | 🔴 Malignant | Keratin pearl formation, intercellular bridges (desmosomes), individual cell keratinisation, markedly pleomorphic nuclei | C34.1 |

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Deep Learning | PyTorch · torchvision · timm · EfficientNet-B0 |
| Training | Two-phase transfer learning · AdamW · ReduceLROnPlateau · Label smoothing · Early stopping · Google Colab T4 GPU |
| Data Pipeline | Custom PyTorch Dataset · Reproducible 85/15 train/val split · MD5 integrity verification · CSV manifest indexing |
| Evaluation | Sensitivity · Specificity · AUC-ROC · PPV · NPV · Confusion matrix · Grad-CAM visualisations |
| LLM Engineering | Anthropic Claude API · Dual prompt paths · Structured JSON output · Graceful fallback handling |
| Frontend | Streamlit · Plotly · Custom CSS (DM Serif Display · DM Mono typography) |
| Deployment | Docker · Hugging Face Spaces · Hugging Face Model Hub |
| Tooling | Git · conda · Jupyter · pandas · NumPy · scikit-learn · Pillow |

---

## Repository Structure

```
patholens/
│
├── app/
│   ├── app.py                ← Streamlit frontend (two-view UX)
│   └── sample_images/        ← 15 curated demo slides (3 per class)
│       ├── colon_aca__*.jpeg
│       ├── colon_n__*.jpeg
│       ├── lung_aca__*.jpeg
│       ├── lung_n__*.jpeg
│       └── lung_scc__*.jpeg
│
├── src/
│   ├── dataset.py            ← PyTorch Dataset class + augmentation pipelines
│   ├── model.py              ← EfficientNet-B0 definition + freeze/unfreeze utilities
│   ├── train.py              ← Two-phase training loop + early stopping
│   ├── evaluate.py           ← Clinical metrics suite + Grad-CAM
│   └── report_generator.py  ← LLM report generation + prompt engineering
│
├── notebooks/
│   ├── 01_EDA.ipynb          ← EDA with clinical annotations
│   └── figures/
│       ├── 01_class_distribution.png
│       ├── 03_sample_image_grid.png
│       ├── 08_training_curves.png
│       ├── 09_confusion_matrix.png
│       ├── 10_roc_curves.png
│       ├── 11_clinical_metrics.png
│       └── 12_gradcam.png
│
├── data/
│   ├── manifest.csv          ← Dataset index (25,000 rows)
│   ├── split_config.json     ← Split reproducibility record
│   └── normalisation_stats.json
│
├── models/
│   └── evaluation_metrics.json
│
├── prepare_dataset.py        ← Data pipeline script
├── Dockerfile                ← HF Spaces container configuration
├── requirements.txt
└── README.md
```

> **Model weights** (`best_model.pth`) are hosted on [Hugging Face Model Hub](https://huggingface.co/T-Letuka/patholens-efficientnet-b0) and downloaded automatically at app startup.

---

## Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/T-Letuka/Patholens.git
cd Patholens

# 2. Create and activate environment
conda create -n patholens python=3.10
conda activate patholens

# 3. Install dependencies
pip install -r requirements.txt

# 4. Model weights are downloaded automatically on first run
#    Or manually: place best_model.pth in models/

# 5. Configure API key
mkdir -p .streamlit
echo 'ANTHROPIC_API_KEY = "sk-ant-your-key-here"' > .streamlit/secrets.toml

# 6. Run
streamlit run app/app.py
```

The app opens at `http://localhost:8501`.

---

## Train From Scratch

```bash
# 1. Download LC25000 from Kaggle
kaggle datasets download -d andrewmvd/lung-and-colon-cancer-histopathological-images

# 2. Unzip into data/raw/
unzip *.zip -d data/raw/

# 3. Prepare dataset (creates manifest + processed splits)
python prepare_dataset.py

# 4. Train (recommended: run on Google Colab with T4 GPU)
python src/train.py

# 5. Evaluate
python src/evaluate.py
```

Training takes approximately 2 hours on a T4 GPU (30 epochs across Phase 1 and Phase 2).

---

## Evaluation Results

### Confusion Matrix
All 2,499 test images correctly classified. Zero off-diagonal entries.

### Per-Class Clinical Metrics

| Class | Sensitivity | Specificity | AUC-ROC | Type |
|-------|------------|------------|---------|------|
| Colon Adenocarcinoma | 100.0% | 100.0% | 1.0000 | 🔴 Malignant |
| Colon Benign | 100.0% | 100.0% | 1.0000 | 🟢 Benign |
| Lung Adenocarcinoma | 100.0% | 100.0% | 1.0000 | 🔴 Malignant |
| Lung Benign | 100.0% | 100.0% | 1.0000 | 🟢 Benign |
| Lung Squamous Cell Carcinoma | 100.0% | 100.0% | 1.0000 | 🔴 Malignant |

### Training Curves
Phase 1 (head only): val accuracy 95.26% → 96.03% over 10 epochs
Phase 2 (full fine-tune): val accuracy 99.61% → 100.00% over 15 epochs

---

## Limitations

PathoLens is a research prototype. The following limitations apply:

- **Benchmark conditions:** LC25000 uses consistent staining, resolution, and acquisition protocols. Real clinical slides show greater variation between laboratories, scanner models, and staining batches. Stain normalisation (e.g. Macenko method) would be required for multi-site deployment.

- **Scope:** Five tissue classes only. Clinical histopathology involves hundreds of diagnostic categories across all organ systems.

- **Patch-level only:** PathoLens classifies 224×224px image patches. Clinical whole-slide images (gigapixel files) require tiling, patch-level inference, and slide-level aggregation (not implemented here).

- **No regulatory approval:** This system has not undergone regulatory review. Clinical deployment would require CE marking (EU/UK) or FDA 510(k) clearance (USA) and prospective clinical validation.

- **Report is preliminary only:** Every generated report includes a mandatory disclaimer stating it must not be used as a standalone diagnostic report and requires pathologist review before clinical action.

---

## Clinical Disclaimer

> PathoLens is a research prototype developed for portfolio and educational purposes. It is **not approved for clinical diagnostic use**. All AI-generated reports are preliminary and must be reviewed by a qualified pathologist before any clinical action is taken. The system has not undergone regulatory review or clinical validation.

---

## Author

**Tisetso Letuka**

BSc Biomedical Science · Honours in Anatomical Pathology (Cum laude) · Data Science

The clinical content in this project  pathological descriptions, ICD-10 codes, microscopic feature annotations, escalation protocols, and augmentation rationale — was authored from domain knowledge, not sourced from tutorials or documentation. The combination of biomedical expertise and end-to-end ML engineering is the point.

- GitHub: [github.com/T-Letuka](https://github.com/T-Letuka)
- Live Demo: https://patholens.streamlit.app/

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>Built with PyTorch · timm · Anthropic API · Streamlit · Streamlit</sub>
</div>
