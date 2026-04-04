import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict

import anthropic

log = logging.getLogger(__name__)

LOW_CONFIDENCE_THRESHOLD = 0.80

CLAUDE_MODEL = "claude-sonnet-4-20250514"

# Maximum tokens for the generated report
MAX_TOKENS = 1024 #noted -  after running a test , the cost for each is 0.03 so 5$ worth of credit is enough.


CLASS_CLINICAL_INFO = {
    "Colon Adenocarcinoma": {
        "icd_code":      "C18.9", #colon specific location was not given hence the code
        "malignant":     True,
        "site":          "colon",
        "description":   (
            "Malignant epithelial tumour of the large intestine arising from "
            "glandular epithelium. Characterised by irregular glandular architecture, "
            "nuclear pleomorphism, loss of cellular polarity, increased mitotic "
            "figures, and desmoplastic stroma."
        ),
        "clinical_significance": (
            "Colorectal adenocarcinoma requires urgent oncological referral. "
            "Standard workup includes staging CT (chest/abdomen/pelvis), "
            "CEA tumour marker, and multidisciplinary team (MDT) discussion. "
            "Treatment options include surgical resection, chemotherapy (FOLFOX/FOLFIRI), "
            "and targeted therapy depending on MSI/MMR status and RAS mutation profile."
        ),
    },
    "Colon Benign": {
        "icd_code":      "K63.9",
        "malignant":     False,
        "site":          "colon",
        "description":   (
            "Normal colonic mucosa with orderly crypt architecture, regular goblet "
            "cells with clear mucin vacuoles, basally oriented uniform nuclei, "
            "and intact basement membrane. No architectural distortion or nuclear atypia."
        ),
        "clinical_significance": (
            "No evidence of malignancy on this biopsy. Clinical correlation with "
            "endoscopic findings and patient history is advised. Routine surveillance "
            "as per local colorectal cancer screening guidelines."
        ),
    },
    "Lung Adenocarcinoma": {
        "icd_code":      "C34.1",
        "malignant":     True,
        "site":          "lung",
        "description":   (
            "Most common primary lung malignancy, arising from peripheral glandular "
            "epithelium. Characterised by acinar, papillary, micropapillary, or lepidic "
            "growth patterns. Features include mucin production, nuclear atypia, "
            "prominent nucleoli, and reactive fibrotic stroma."
        ),
        "clinical_significance": (
            "Lung adenocarcinoma requires urgent respiratory oncology referral. "
            "Staging PET-CT and brain MRI are standard. Molecular profiling is "
            "mandatory (EGFR, ALK, ROS1, PD-L1, KRAS) to guide targeted therapy "
            "eligibility. MDT discussion for surgical, chemotherapy, immunotherapy, "
            "or combined modality treatment planning."
        ),
    },
    "Lung Benign": {
        "icd_code":      "J98.4",
        "malignant":     False,
        "site":          "lung",
        "description":   (
            "Normal lung parenchyma with patent alveolar spaces lined by flat type I "
            "pneumocytes, cuboidal type II pneumocytes at alveolar corners, uniform "
            "alveolar wall thickness, and intact capillary network. "
            "No evidence of cellular atypia, fibrosis, or architectural distortion."
        ),
        "clinical_significance": (
            "No evidence of malignancy on this specimen. Clinical correlation with "
            "radiological findings and patient history recommended. "
            "If symptoms persist, consider further investigation per respiratory "
            "medicine guidelines."
        ),
    },
    "Lung Squamous Cell Carcinoma": {
        "icd_code":      "C34.1",
        "malignant":     True,
        "site":          "lung",
        "description":   (
            "Malignant tumour arising from squamous metaplasia of bronchial epithelium. "
            "Characterised by keratin pearl formation (concentric whorls of keratinising "
            "cells), intercellular bridges (desmosomes), individual cell keratinisation "
            "with bright eosinophilic cytoplasm, and markedly pleomorphic hyperchromatic nuclei."
        ),
        "clinical_significance": (
            "Lung squamous cell carcinoma requires urgent respiratory oncology referral. "
            "Staging PET-CT and brain MRI recommended. PD-L1 expression testing is "
            "standard for immunotherapy eligibility. EGFR mutations are rare in SCC — "
            "targeted therapy is less commonly applicable. MDT discussion for surgery, "
            "chemoradiotherapy, or immunotherapy planning."
        ),
    },
}


# -----------------------------------------------------------------------------
# PREDICTION DATA STRUCTURE
# -----------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """
    Structured container for CNN prediction output.

    This is passed to the report generator functions.
    Using a dataclass instead of a plain dict enforces the expected
    structure ,  if a field is missing, Python raises an error immediately
    rather than silently producing a malformed report.

    Attributes:
        predicted_class:  The class with the highest probability
        confidence:       Probability of the predicted class (0.0 - 1.0)
        all_probabilities: Dict mapping each class name to its probability
        image_filename:   Original image filename (for the report header)
    """
    predicted_class:    str
    confidence:         float
    all_probabilities:  dict
    image_filename:     str = "uploaded_image.jpg"

    @property
    def is_low_confidence(self) -> bool:
        return self.confidence < LOW_CONFIDENCE_THRESHOLD

    @property
    def is_malignant(self) -> bool:
        return CLASS_CLINICAL_INFO[self.predicted_class]["malignant"]

    @property
    def top_3_differentials(self) -> list:
        """Returns top 3 classes sorted by probability, as list of (class, prob) tuples."""
        sorted_probs = sorted(
            self.all_probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_probs[:3]

    def to_dict(self) -> dict:
        return asdict(self)


# -----------------------------------------------------------------------------
# SYSTEM PROMPT
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a pathology reporting assistant for PathoLens, an AI-assisted 
histopathology image analysis system. You generate structured preliminary pathology reports 
based on deep learning classifier predictions from digitised H&E stained tissue slides.

CRITICAL CONSTRAINTS:
1. You are generating a PRELIMINARY AI-ASSISTED report, NOT a diagnostic report.
2. Always include the mandatory disclaimer at the end of every report.
3. Use formal pathology reporting language throughout.
4. Never express absolute certainty — use appropriate hedging language 
   ("consistent with", "features suggestive of", "findings are in keeping with").
5. For malignant predictions, always include escalation language and next steps.
6. For low-confidence predictions, never state a primary diagnosis — 
   list differentials only and flag for expert review.
7. Keep reports concise and structured — pathologists read many reports per day.

OUTPUT FORMAT:
Always structure your response as valid JSON with these exact keys:
{
  "report_type": "high_confidence" or "low_confidence",
  "header": { ... },
  "microscopic_description": "...",
  "interpretation": "...", 
  "clinical_significance": "...",
  "recommended_next_steps": "...",
  "disclaimer": "..."
}

Do not include any text outside the JSON object."""


# -----------------------------------------------------------------------------
# HIGH CONFIDENCE PROMPT
# -----------------------------------------------------------------------------

def build_high_confidence_prompt(prediction: PredictionResult) -> str:
    """
    Builds the user message for a high-confidence prediction.

    Provides the model with:
    - The predicted class and confidence
    - All class probabilities (for context)
    - Clinical information about the predicted class
    - Specific instructions for the 4-section report
    """
    info = CLASS_CLINICAL_INFO[prediction.predicted_class]

    # Format probability table for the prompt
    prob_table = "\n".join([
        f"  {cls:<35} {prob*100:>6.2f}%"
        for cls, prob in sorted(
            prediction.all_probabilities.items(),
            key=lambda x: x[1], reverse=True
        )
    ])

    malignant_flag = "MALIGNANT" if info["malignant"] else "BENIGN"

    return f"""Generate a preliminary pathology report for the following AI classifier prediction.

CLASSIFIER OUTPUT:
  Image filename    : {prediction.image_filename}
  Predicted class   : {prediction.predicted_class}
  Classification    : {malignant_flag}
  Confidence        : {prediction.confidence*100:.1f}%
  Tissue site       : {info['site'].upper()}

ALL CLASS PROBABILITIES:
{prob_table}

REFERENCE PATHOLOGY (for context):
  Histological description : {info['description']}
  Clinical significance    : {info['clinical_significance']}
  ICD-10 code              : {info['icd_code']}

REPORT INSTRUCTIONS:
Generate a JSON report with these sections:

1. header: Include specimen_id (use filename), date (today), classifier_version ("EfficientNet-B0 v1.0"), confidence_level ("{prediction.confidence*100:.1f}%"), classification ("{malignant_flag}")

2. microscopic_description: Describe the tissue morphology consistent with {prediction.predicted_class}. 
   Use formal pathology language. Reference the key histological features that define this diagnosis.
   Write as if describing what would be seen on the slide — not what the AI saw.

3. interpretation: State the AI-assisted impression. Use hedging language ("features are consistent with", 
   "findings are in keeping with"). Include the confidence percentage.
   {"Include a note that malignancy is suspected and urgent clinical correlation is required." if info['malignant'] else "State that no features of malignancy are identified on this specimen."}

4. clinical_significance: Explain what this finding means for the patient's clinical pathway.
   {"Include MDT discussion, staging investigations, and specialist referral language." if info['malignant'] else "Include routine surveillance and clinical correlation language."}

5. recommended_next_steps: List 3-5 specific clinical actions appropriate for this diagnosis.
   {"Include: urgent oncology referral, staging imaging, molecular profiling where applicable." if info['malignant'] else "Include: clinical correlation, routine follow-up, further investigation if symptomatic."}

6. disclaimer: Use exactly this text: "IMPORTANT: This report is generated by an AI-assisted image 
   analysis system (PathoLens) and constitutes a PRELIMINARY SCREENING AID ONLY. It must not be 
   used as a standalone diagnostic report. All findings require review and authorisation by a 
   qualified pathologist before clinical action is taken. This system has not received regulatory 
   approval for clinical diagnostic use."

Respond with valid JSON only. No text outside the JSON object."""


# -----------------------------------------------------------------------------
# LOW CONFIDENCE PROMPT
# -----------------------------------------------------------------------------

def build_low_confidence_prompt(prediction: PredictionResult) -> str:
    """
    Builds the user message for a low-confidence prediction.

    Key differences from high-confidence:
    - No primary diagnosis is stated
    - Differentials are listed with probabilities
    - Mandatory expert review flag
    - More conservative language throughout
    """
    prob_table = "\n".join([
        f"  {cls:<35} {prob*100:>6.2f}%"
        for cls, prob in sorted(
            prediction.all_probabilities.items(),
            key=lambda x: x[1], reverse=True
        )
    ])

    top3 = prediction.top_3_differentials
    differentials_text = "\n".join([
        f"  {i+1}. {cls} ({prob*100:.1f}%)"
        for i, (cls, prob) in enumerate(top3)
    ])

    return f"""Generate a preliminary pathology report for the following LOW-CONFIDENCE AI classifier output.

CLASSIFIER OUTPUT:
  Image filename    : {prediction.image_filename}
  Top prediction    : {prediction.predicted_class}
  Confidence        : {prediction.confidence*100:.1f}%  ← BELOW 80% THRESHOLD
  Status            : REQUIRES EXPERT REVIEW

ALL CLASS PROBABILITIES:
{prob_table}

TOP 3 DIFFERENTIALS:
{differentials_text}

REPORT INSTRUCTIONS — LOW CONFIDENCE MODE:
This case has insufficient classifier confidence for a primary AI diagnosis.
Generate a JSON report with these sections:

1. header: Include specimen_id (use filename), date (today), classifier_version ("EfficientNet-B0 v1.0"), 
   confidence_level ("{prediction.confidence*100:.1f}% — INDETERMINATE"), 
   classification ("INDETERMINATE — EXPERT REVIEW REQUIRED"),
   alert ("⚠ LOW CONFIDENCE PREDICTION — DO NOT USE FOR CLINICAL DECISION MAKING")

2. microscopic_description: Describe the tissue features that may be present given the differential 
   diagnoses listed. Use tentative language. Note that the classifier found overlapping features 
   between multiple classes.

3. interpretation: State that the AI classifier has returned an INDETERMINATE result with 
   confidence below the 80% clinical threshold. List the top 3 differential diagnoses with 
   their probabilities. Do NOT state a primary diagnosis. Explicitly recommend pathologist review.

4. clinical_significance: Explain that no clinical decisions should be made based on this result. 
   State that the ambiguity may reflect genuine tissue complexity, slide quality issues, 
   or a rare/unusual presentation requiring expert interpretation.

5. recommended_next_steps: 
   - Mandatory pathologist review of original slide
   - Consider additional immunohistochemical stains if appropriate
   - Clinical correlation with radiological and endoscopic findings
   - Re-submission with higher quality slide preparation if applicable

6. disclaimer: Use exactly this text: "IMPORTANT: This report is generated by an AI-assisted image 
   analysis system (PathoLens) and constitutes a PRELIMINARY SCREENING AID ONLY. It must not be 
   used as a standalone diagnostic report. All findings require review and authorisation by a 
   qualified pathologist before clinical action is taken. This system has not received regulatory 
   approval for clinical diagnostic use. THIS CASE HAS BEEN FLAGGED AS INDETERMINATE AND REQUIRES 
   MANDATORY EXPERT REVIEW BEFORE ANY CLINICAL ACTION."

Respond with valid JSON only. No text outside the JSON object."""


# -----------------------------------------------------------------------------
# REPORT GENERATOR
# -----------------------------------------------------------------------------

def generate_report(
    prediction:  PredictionResult,
    api_key:     str | None = None,
) -> dict:
    """
    Main entry point. Generates a clinical report for a prediction.

    Args:
        prediction: PredictionResult from the CNN classifier
        api_key:    Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.

    Returns:
        A dict with the structured report fields.
        Always includes a 'report_type' key: "high_confidence" or "low_confidence"
        Always includes a 'generation_metadata' key with model info.

    Raises:
        ValueError: if the API key is missing
        anthropic.APIError: if the API call fails
    """

    # Resolve API key
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError(
            "Anthropic API key not found. "
            "Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
        )

    client = anthropic.Anthropic(api_key=key)

    # Select the appropriate prompt based on confidence
    if prediction.is_low_confidence:
        user_prompt = build_low_confidence_prompt(prediction)
        log.info(
            f"LOW CONFIDENCE path: {prediction.predicted_class} "
            f"({prediction.confidence*100:.1f}%) — generating differential report"
        )
    else:
        user_prompt = build_high_confidence_prompt(prediction)
        log.info(
            f"HIGH CONFIDENCE path: {prediction.predicted_class} "
            f"({prediction.confidence*100:.1f}%) — generating primary diagnosis report"
        )

    # Call the API
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_prompt}
        ],
    )

    raw_text = response.content[0].text.strip()

 
    if raw_text.startswith("```"):
        lines    = raw_text.split("\n")
        raw_text = "\n".join(lines[1:-1])

    try:
        report = json.loads(raw_text)
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse report JSON: {e}")
        log.error(f"Raw response: {raw_text[:500]}")
        # Return a safe fallback rather than crashing the app
        report = _fallback_report(prediction)

    # Attach generation metadata
    # This is displayed in the Streamlit app's About panel
    report["generation_metadata"] = {
        "model":            CLAUDE_MODEL,
        "predicted_class":  prediction.predicted_class,
        "confidence":       round(prediction.confidence, 4),
        "low_confidence":   prediction.is_low_confidence,
        "threshold_used":   LOW_CONFIDENCE_THRESHOLD,
        "input_tokens":     response.usage.input_tokens,
        "output_tokens":    response.usage.output_tokens,
    }

    return report


def _fallback_report(prediction: PredictionResult) -> dict:
    """
    Returns a minimal safe report if JSON parsing fails.
    Ensures the Streamlit app never crashes due to a malformed API response.
    """
    return {
        "report_type": "error",
        "header": {
            "specimen_id":        prediction.image_filename,
            "classifier_version": "EfficientNet-B0 v1.0",
            "confidence_level":   f"{prediction.confidence*100:.1f}%",
            "classification":     "ERROR — SEE BELOW",
        },
        "microscopic_description": "Report generation encountered an error.",
        "interpretation":          f"Top prediction: {prediction.predicted_class} ({prediction.confidence*100:.1f}%)",
        "clinical_significance":   "Please consult a qualified pathologist.",
        "recommended_next_steps":  "Manual review required.",
        "disclaimer":              (
            "IMPORTANT: This report is generated by an AI-assisted system and must not "
            "be used as a standalone diagnostic report."
        ),
    }


# -----------------------------------------------------------------------------
# REPORT FORMATTER
# -----------------------------------------------------------------------------

def format_report_as_markdown(report: dict) -> str:
    """
    Converts the structured report dict into formatted markdown for display
    in the Streamlit app.

    The markdown is rendered directly by st.markdown() in app.py.
    """
    header = report.get("header", {})
    meta   = report.get("generation_metadata", {})
    is_low = report.get("report_type") == "low_confidence"
    is_mal = meta.get("predicted_class") in [
        k for k, v in CLASS_CLINICAL_INFO.items() if v["malignant"]
    ]

    # Alert banner for low confidence or malignant predictions
    if is_low:
        banner = "⚠️ **INDETERMINATE RESULT — EXPERT REVIEW REQUIRED**"
        banner_color = "🟡"
    elif is_mal:
        banner = "🔴 **MALIGNANT TISSUE IDENTIFIED — URGENT CLINICAL CORRELATION REQUIRED**"
        banner_color = "🔴"
    else:
        banner = "🟢 **NO EVIDENCE OF MALIGNANCY ON THIS SPECIMEN**"
        banner_color = "🟢"

    md = f"""---
## {banner_color} PathoLens Preliminary Pathology Report

> {banner}

---

### 📋 Report Header

| Field | Value |
|-------|-------|
| Specimen ID | `{header.get('specimen_id', 'N/A')}` |
| Classification | **{header.get('classification', 'N/A')}** |
| Confidence | {header.get('confidence_level', 'N/A')} |
| Classifier | {header.get('classifier_version', 'N/A')} |
| Date | {header.get('date', 'N/A')} |

---

### 🔬 Microscopic Description

{report.get('microscopic_description', 'Not available.')}

---

### 📊 Interpretation

{report.get('interpretation', 'Not available.')}

---

### 🏥 Clinical Significance

{report.get('clinical_significance', 'Not available.')}

---

### ✅ Recommended Next Steps

{report.get('recommended_next_steps', 'Not available.')}

---

### ⚠️ Disclaimer

> {report.get('disclaimer', '')}

---
*Generated by PathoLens | Model: {meta.get('model', 'N/A')} | Tokens used: {meta.get('input_tokens', 0) + meta.get('output_tokens', 0)}*
"""
    return md


# -----------------------------------------------------------------------------
# CONVENIENCE FUNCTION FOR STREAMLIT APP
# -----------------------------------------------------------------------------

def predict_and_report(
    class_probabilities: dict,
    image_filename:      str = "uploaded_image.jpg",
    api_key:             str | None = None,
) -> tuple[dict, str]:
    """
    Single function called by the Streamlit app.

    Takes the raw probability dict from the CNN and returns
    both the structured report dict and formatted markdown.

    Args:
        class_probabilities: {class_name: probability} from model inference
        image_filename:      name of the uploaded file
        api_key:             Anthropic API key

    Returns:
        (report_dict, markdown_string)
    """
    # Find predicted class and confidence
    predicted_class = max(class_probabilities, key=class_probabilities.get)
    confidence      = class_probabilities[predicted_class]

    prediction = PredictionResult(
        predicted_class   = predicted_class,
        confidence        = confidence,
        all_probabilities = class_probabilities,
        image_filename    = image_filename,
    )

    report   = generate_report(prediction, api_key=api_key)
    markdown = format_report_as_markdown(report)

    return report, markdown




if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    print("=" * 60)
    print("  PathoLens — Report Generator Test")
    print("=" * 60)
    print()

    # ── TEST 1: High confidence malignant ────────────────────────────────────
    print("TEST 1: High confidence malignant prediction")
    print("-" * 45)

    fake_high_conf = PredictionResult(
        predicted_class   = "Lung Squamous Cell Carcinoma",
        confidence        = 0.962,
        all_probabilities = {
            "Colon Adenocarcinoma":         0.005,
            "Colon Benign":                 0.003,
            "Lung Adenocarcinoma":          0.021,
            "Lung Benign":                  0.009,
            "Lung Squamous Cell Carcinoma": 0.962,
        },
        image_filename = "test_lung_scc_001.jpeg"
    )

    print(f"  Class      : {fake_high_conf.predicted_class}")
    print(f"  Confidence : {fake_high_conf.confidence*100:.1f}%")
    print(f"  Malignant  : {fake_high_conf.is_malignant}")
    print(f"  Low conf?  : {fake_high_conf.is_low_confidence}")
    print()

    try:
        report1, markdown1 = predict_and_report(
            class_probabilities = fake_high_conf.all_probabilities,
            image_filename      = fake_high_conf.image_filename,
        )
        print("Report generated successfully.")
        print(f"  Report type : {report1.get('report_type')}")
        print(f"  Tokens used : {report1['generation_metadata']['input_tokens'] + report1['generation_metadata']['output_tokens']}")
        print()
        print("── MARKDOWN PREVIEW (first 500 chars) ──")
        print(markdown1[:500])
        print("...")

    except Exception as e:
        print(f"Error: {e}")

    print()

    # ── TEST 2: Low confidence ────────────────────────────────────────────────
    print("TEST 2: Low confidence prediction (ambiguous case)")
    print("-" * 45)

    fake_low_conf = PredictionResult(
        predicted_class   = "Lung Adenocarcinoma",
        confidence        = 0.52,
        all_probabilities = {
            "Colon Adenocarcinoma":         0.03,
            "Colon Benign":                 0.05,
            "Lung Adenocarcinoma":          0.52,
            "Lung Benign":                  0.12,
            "Lung Squamous Cell Carcinoma": 0.28,
        },
        image_filename = "ambiguous_lung_case_042.jpeg"
    )

    print(f"  Class      : {fake_low_conf.predicted_class}")
    print(f"  Confidence : {fake_low_conf.confidence*100:.1f}%")
    print(f"  Low conf?  : {fake_low_conf.is_low_confidence}")
    print()

    try:
        report2, markdown2 = predict_and_report(
            class_probabilities = fake_low_conf.all_probabilities,
            image_filename      = fake_low_conf.image_filename,
        )
        print("Report generated successfully.")
        print(f"  Report type : {report2.get('report_type')}")
        print()
        print("── MARKDOWN PREVIEW (first 500 chars) ──")
        print(markdown2[:500])
        print("...")

    except Exception as e:
        print(f"Error: {e}")

    print()
    print("=" * 60)
    print("  Tests complete.")
    print("=" * 60)