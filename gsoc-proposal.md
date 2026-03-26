# GSoC 2026 Contributor Project Proposal
## MedVision + Explainability — HiperHealth

---

## Personal Information

| Field | Details |
|---|---|
| **Full Name** | {{ YOUR_FULL_NAME }} |
| **Email** | {{ YOUR_EMAIL }} |
| **GitHub Username** | {{ YOUR_GITHUB_USERNAME }} |
| **Discord Handle** | {{ YOUR_DISCORD_HANDLE }} |
| **Time Zone** | {{ YOUR_TIMEZONE, e.g. UTC+5:30 }} |
| **Country** | {{ YOUR_COUNTRY }} |
| **University** | {{ YOUR_UNIVERSITY }} |
| **Degree / Year** | {{ e.g. B.Tech Computer Science, 3rd Year }} |
| **LinkedIn** | {{ YOUR_LINKEDIN_URL }} |

---

## Previous Contributions to the Project

I began studying the HiperHealth codebase through the documentation at
`docs/skills.md`, `docs/llm_configuration.md`, and `DEVELOPMENT.md` before
making any code changes. This let me understand the skill lifecycle (`pre` →
`execute` → `post`), the `PipelineContext` serialisation model, and how the
existing `PrivacySkill`, `ExtractionSkill`, and `DiagnosticsSkill` are
structured.

{{ Describe your contributions here. Below is the format OSL requires — fill
in your actual PRs and issues. }}

### Pull Requests

| Title | PR # | Link | Status |
|---|---|---|---|
| {{ PR title }} | {{ #N }} | {{ URL }} | {{ merged / open }} |
| {{ PR title }} | {{ #N }} | {{ URL }} | {{ merged / open }} |

### Issues Resolved / Opened

| Title | Issue # | Link | Status |
|---|---|---|---|
| {{ Issue title }} | {{ #N }} | {{ URL }} | {{ closed / open }} |

---

## Why This Project?

Clinical AI systems today can read lab values and generate differential
diagnoses from text, but the visual channel — what a clinician sees with their
eyes — remains largely unautomated in open-source tooling. Patients photograph
suspicious skin lesions, discoloured nails, inflamed eyes, or a meal plate
before a telehealth call, yet most pipelines discard those images or treat them
as unstructured attachments.

HiperHealth already solves the text side of the problem elegantly: a
composable, stage-independent skill pipeline with FHIR-native outputs and
strong privacy controls. What it is missing is a skill that can turn a JPEG
into a structured, explainable clinical observation — one that a physician can
audit, override, and build on.

The MedVision project matters to me because it sits at the intersection of the
two things I find most technically interesting: computer vision and clinical
standards interoperability (FHIR / SNOMED CT). I have spent the last year
working through papers on Grad-CAM, uncertainty quantification in medical
imaging, and FHIR R4 resource modelling, and I want to apply that background
somewhere it will actually reach patients. HiperHealth's open-source, BSD-3
licence and its active Discord community give me confidence that the work will
be maintained and extended well after GSoC ends.

I also appreciate that the project does not ask for black-box AI. The explicit
requirement for saliency maps and physician-facing dashboards aligns with my
view that clinical AI tools are only trustworthy when they can justify their
outputs in terms a clinician can evaluate.

---

## Project Timeline / Deliverables

> **Total hours:** 350 | **Duration:** ~12 weeks (May 26 – August 25, 2026)
>
> Each phase ends with a working, tested, and documented increment that can be
> reviewed independently.

### Phase 0 — Community Bonding (May 1 – May 25, pre-coding)

| Dates | Work |
|---|---|
| May 1 – May 10 | Deep-read all existing skill implementations (`privacy/deidentifier.py`, `extraction/skill.py`, `diagnostics/core.py`). Run the full test suite locally. Understand how `PipelineContext.extras` is used for skill-to-skill communication. |
| May 11 – May 20 | Discuss architecture decisions with mentors on Discord. Agree on image storage strategy (filesystem paths in context vs. base64 blobs). Confirm FHIR resource schema for visual observations. |
| May 21 – May 25 | Set up experiment tracking (MLflow or W&B, locally). Write the skeleton `hiperhealth.yaml` for both `VisionSkill` and `NutritionSkill`. |

---

### Phase 1 — Image Quality Gate & Privacy Layer (Week 1–2, ~55 h)

**Goal:** No image enters the classification pipeline unless it passes quality
checks and has faces blurred.

| Dates | Deliverables |
|---|---|
| May 26 – Jun 1 | `src/hiperhealth/skills/vision/quality.py` — `ImageQualityChecker` class: blur score (Laplacian variance), brightness histogram check, minimum resolution enforcement. Returns a `QualityReport` dataclass with `passed: bool` and `reason: str`. |
| Jun 2 – Jun 8 | `src/hiperhealth/skills/vision/privacy.py` — `ImagePrivacyFilter`: wraps OpenCV's DNN face detector to blur facial regions before any model inference. Integrates with the existing `PrivacySkill` by running in the `pre()` hook of `VisionSkill` during the `SCREENING` stage. |
| Jun 8 (milestone) | `tests/test_vision_quality.py` and `tests/test_vision_privacy.py` with ≥ 92 % coverage. PR up for mentor review. |

**Key design decision:** The `VisionSkill.pre()` in `SCREENING` runs the face
blur and quality check. If quality fails, it sets
`ctx.extras["vision_skip"] = True` and logs an `AuditEntry`. The
`execute()` hook checks this flag and returns early, so the skill is
transparently skippable without raising an exception.

---

### Phase 2 — Body-Region Classifiers (Week 3–5, ~75 h)

**Goal:** Five fine-tuned ViT-B/16 models (eyes, nails, skin, tongue, ears)
that return a predicted condition label with a calibrated confidence score.

| Dates | Deliverables |
|---|---|
| Jun 9 – Jun 15 | Data pipeline: download and preprocess HAM10000 (skin), Ocular Disease Recognition dataset (eyes), and community-curated nail/tongue/ear datasets. Write `scripts/prepare_vision_datasets.py`. |
| Jun 16 – Jun 22 | Transfer learning loop using HuggingFace `transformers` — `ViTForImageClassification` fine-tuned per region. Use temperature scaling post-training for calibration (Platt scaling). |
| Jun 23 – Jun 29 | Export five ONNX models to `src/hiperhealth/skills/vision/models/`. Write `ModelRegistry` (a simple dict keyed by region name) loaded lazily on first use to keep import time low. |
| Jun 30 (milestone) | Each model achieves ≥ 80 % top-1 accuracy on held-out validation split. Benchmark report committed to `docs/vision_benchmarks.md`. |
| Jul 1 – Jul 6 | `VisionSkill.execute()` in `INTAKE` stage: detects the body region from filename metadata or a lightweight ResNet-18 region detector, routes to the appropriate model, and returns `VisionResult` stored in `ctx.results["vision"]`. |

**`VisionResult` schema (Pydantic):**

```python
class RegionPrediction(BaseModel):
    region: str          # "skin" | "eye" | "nail" | "tongue" | "ear"
    label: str           # ICD-11 concept label
    icd11_code: str      # e.g. "EA90" for Atopic dermatitis
    confidence: float    # calibrated probability [0, 1]
    uncertainty: float   # MC-Dropout std dev over T=20 passes

class VisionResult(BaseModel):
    predictions: list[RegionPrediction]
    image_path: str
    processed_at: str    # ISO-8601 timestamp
```

---

### Phase 3 — Explainability Module (Week 6–7, ~55 h)

**Goal:** Every prediction ships with a spatial saliency map (Grad-CAM) and
a per-feature attribution (SHAP) that physicians can inspect.

| Dates | Deliverables |
|---|---|
| Jul 7 – Jul 13 | `src/hiperhealth/skills/vision/explainability.py` — `GradCAMExplainer`: computes Grad-CAM heatmap for the target class using the last convolutional block of the ViT (patch-level gradients via `pytorch-grad-cam`). Outputs a PNG overlay saved alongside the source image. |
| Jul 14 – Jul 20 | `SHAPExplainer`: uses `shap.DeepExplainer` (faster than `KernelExplainer` for neural nets) to produce superpixel-level attributions. Outputs a JSON array of `(region_name, shap_value)` pairs. `UncertaintyEstimator`: wraps the ONNX model to enable MC-Dropout via repeated inference with dropout enabled. |
| Jul 20 (milestone) | `tests/test_explainability.py` with mock models. Saliency maps generated on the five test images in `tests/data/clinical_images/`. PR reviewed by mentors. |

**Integration:** `VisionSkill.post()` calls both explainers and attaches outputs
to `ctx.extras["vision_explainability"]`:

```python
ctx.extras["vision_explainability"] = {
    "gradcam_path": "/tmp/hh_gradcam_abc123.png",
    "shap_values": [...],
    "uncertainty": 0.08
}
```

---

### Phase 4 — Nutritional Estimation Pipeline (Week 8–9, ~55 h)

**Goal:** Meal photos produce structured nutritional estimates (±20 % calorie
accuracy, full macronutrient breakdown).

| Dates | Deliverables |
|---|---|
| Jul 21 – Jul 27 | `NutritionSkill` registered for the `INTAKE` stage. `FoodDetector`: fine-tune YOLOv8-nano on FOOD-101 + UEC Food-256. Identifies individual food items and bounding boxes. |
| Jul 28 – Aug 3 | `NutritionEstimator`: maps YOLO detections to the USDA FoodData Central API (free, open REST API) by food name. Applies portion size heuristics based on bounding-box relative area and known plate-size priors. Returns `NutritionResult` stored in `ctx.results["nutrition"]`. |
| Aug 3 (milestone) | Calorie estimation on a held-out 200-image test set achieves ≤ 20 % MAPE. Macro breakdown (protein, fat, carbohydrates) validated against ground-truth nutrition labels. |

**`NutritionResult` schema:**

```python
class FoodItem(BaseModel):
    name: str
    usda_fdc_id: str
    calories_kcal: float
    protein_g: float
    fat_g: float
    carbs_g: float
    confidence: float

class NutritionResult(BaseModel):
    items: list[FoodItem]
    total_calories_kcal: float
    total_protein_g: float
    total_fat_g: float
    total_carbs_g: float
    image_path: str
```

---

### Phase 5 — FHIR Schema Extensions & Pipeline Integration (Week 10, ~30 h)

**Goal:** Both `VisionResult` and `NutritionResult` serialise to valid FHIR R4
`Observation` resources so that downstream skills (DiagnosticsSkill) and the
`hiperhealth-web` application can consume them without schema changes.

| Dates | Deliverables |
|---|---|
| Aug 4 – Aug 10 | Add `VisualObservation` and `NutritionalObservation` to `src/hiperhealth/schema/fhirx.py`, extending the existing `Observation(FhirObservation, BaseLanguage)` pattern already used in the codebase. |

**`VisualObservation` extension:**

```python
class VisualObservation(Observation):
    icd11_code: Optional[str] = Field(None, description="ICD-11 concept code")
    confidence: Optional[float] = Field(None, description="Model confidence [0,1]")
    uncertainty: Optional[float] = Field(None, description="MC-Dropout std dev")
    saliency_map_url: Optional[str] = Field(None, description="URL or path to Grad-CAM PNG")
    shap_summary: Optional[list[dict]] = Field(None, description="Top-5 SHAP attributions")
    body_region: Optional[str] = Field(None, description="Examined body region")
```

**`NutritionalObservation` extension:**

```python
class NutritionalObservation(Observation):
    total_calories_kcal: Optional[float] = None
    total_protein_g: Optional[float] = None
    total_fat_g: Optional[float] = None
    total_carbs_g: Optional[float] = None
    food_items: Optional[list[dict]] = None
```

| Dates | Deliverables |
|---|---|
| Aug 4 – Aug 10 (cont.) | Run `makim gen.fhir-models` to regenerate `src/hiperhealth/models/sqla/fhirx.py` with the new columns. Update `hiperhealth.yaml` manifests for both skills with correct stages, versions, and entry points. Wire `VisionSkill` into `create_default_runner()` alongside the existing three skills. |

---

### Phase 6 — Physician Feedback & Audit Integration (Week 11, ~25 h)

**Goal:** Physicians can flag incorrect AI predictions; corrections are stored
as FHIR `Annotation` resources and logged in the existing audit trail.

| Dates | Deliverables |
|---|---|
| Aug 11 – Aug 17 | `VisionSkill.post()` appends an `AuditEntry` (already used by all built-in skills) recording the model name, version, confidence, and uncertainty. A `PhysicianFeedback` Pydantic model captures the corrected label and the physician's free-text note. This is serialised as a FHIR `Annotation` and appended to `ctx.extras["physician_annotations"]`. The feedback structure is intentionally stored separately so a future fine-tuning loop (outside GSoC scope) can pick it up. |

---

### Phase 7 — Tests, Documentation & Final Polish (Week 12, ~55 h)

**Goal:** Every new module ships with comprehensive tests, documentation, and
an end-to-end usage example.

| Dates | Deliverables |
|---|---|
| Aug 18 – Aug 22 | Comprehensive test suite for all new modules: `test_vision_skill.py`, `test_nutrition_skill.py`, `test_fhirx_visual.py`, `test_fhirx_nutrition.py`. Integration test runs the full pipeline end-to-end on a synthetic patient record with attached clinical image and meal photo. Achieve ≥ 92 % coverage (matching the project's existing standard). |
| Aug 22 – Aug 24 | Documentation: `docs/vision_skill.md` (usage guide, model card, environment requirements), `docs/nutrition_skill.md`, updated `docs/skills.md` with new examples. Update `mkdocs.yaml` navigation. |
| Aug 25 (final) | Clean up all open PRs. Tag release candidate. Write the final GSoC report blog post. |

---

### Summary of Deliverables

| # | Deliverable | Stage |
|---|---|---|
| 1 | `ImageQualityChecker` — blur, brightness, resolution validation | Phase 1 |
| 2 | `ImagePrivacyFilter` — face detection + blur (OpenCV DNN) | Phase 1 |
| 3 | Five fine-tuned ViT-B/16 body-region classifiers (ONNX) | Phase 2 |
| 4 | `VisionSkill` — `pre/execute/post` for `SCREENING` + `INTAKE` stages | Phase 2 |
| 5 | `GradCAMExplainer` — spatial saliency maps per prediction | Phase 3 |
| 6 | `SHAPExplainer` — superpixel feature attributions | Phase 3 |
| 7 | `UncertaintyEstimator` — MC-Dropout confidence intervals | Phase 3 |
| 8 | `FoodDetector` — YOLOv8-nano fine-tuned on FOOD-101 + UEC | Phase 4 |
| 9 | `NutritionEstimator` — USDA FoodData Central integration | Phase 4 |
| 10 | `NutritionSkill` — `pre/execute/post` for `INTAKE` stage | Phase 4 |
| 11 | `VisualObservation` FHIR schema + SQLAlchemy model | Phase 5 |
| 12 | `NutritionalObservation` FHIR schema + SQLAlchemy model | Phase 5 |
| 13 | Physician feedback model (`PhysicianFeedback` + FHIR `Annotation` integration) | Phase 6 |
| 14 | Full test suite (≥ 92 % coverage) + integration test | Phase 7 |
| 15 | Complete documentation for both skills | Phase 7 |

---

## Availability

I am available full-time during the GSoC coding period (May 26 – August 25,
2026).

- **Weekly hours:** 35–40 hours per week.
- **Other commitments:** {{ List any exams, part-time work, travel, etc. If
  none, write "No other commitments during this period." }}
- **Catch-up plan:** I have deliberately front-loaded the more research-heavy
  work (dataset preparation, model training) in Phases 1–4 and reserved Phase
  7 as a buffer. If any phase runs behind, I will raise it in the weekly
  mentor check-in immediately rather than trying to silently make up time. The
  physician feedback feature (Phase 6) can be scoped down to a simpler JSON
  log if schedule pressure requires it, without affecting the core deliverables.
- **Communication:** I will post a short progress note in the HiperHealth
  Discord channel every Friday and open a draft PR for each phase milestone so
  mentors can give early feedback.

---

## Blog Posts

I commit to writing at least **one public blog post per phase** throughout the
coding period — seven posts in total. Each post will cover what I built, what
I learned (including dead ends), and how the work fits into HiperHealth's
broader clinical AI pipeline. I plan to publish on {{ your blog platform —
e.g. dev.to, Hashnode, personal site }}.

Planned post titles:

1. *"Building a Privacy-First Image Intake for Clinical AI"* (Phase 1)
2. *"Fine-Tuning Vision Transformers for Multi-Region Clinical Classification"* (Phase 2)
3. *"Making Medical AI Explainable: Grad-CAM, SHAP, and MC-Dropout in Practice"* (Phase 3)
4. *"From Meal Photo to Macros: Open-Source Nutritional Estimation"* (Phase 4)
5. *"Bridging Deep Learning and FHIR R4: VisualObservation Resources"* (Phase 5)
6. *"Closing the Loop: Physician Feedback in AI-Augmented Clinical Workflows"* (Phase 6)
7. *"GSoC 2026 Final Report: What MedVision Added to HiperHealth"* (Phase 7)

---

## Post-GSoC

I intend to remain an active contributor to HiperHealth after the programme
ends. Concretely:

- **Model maintenance:** Re-train the classifiers when higher-quality public
  datasets become available (e.g. ISIC 2026 challenge data).
- **DICOM support:** The current pipeline processes JPEG/PNG. Adding DICOM
  ingestion via `pydicom` is a natural next step for institutional adoption.
- **Federated inference:** HiperHealth's stateless, skill-based design is a
  good fit for federated scenarios where images never leave the hospital.
  I want to prototype a privacy-preserving variant using embedding-only
  transmission.
- **Community engagement:** Answer questions on Discord, review PRs, and help
  onboard future contributors who want to extend the vision module.

---

## Technical Background

### Relevant Skills

- **Python:** 3+ years. Comfortable with type annotations, Pydantic v2, async
  patterns, and packaging with `pyproject.toml`.
- **Computer Vision:** PyTorch, torchvision, HuggingFace `transformers`
  (ViT, Swin). Familiar with Grad-CAM via `pytorch-grad-cam`, SHAP for
  deep networks, and ONNX export/inference.
- **FHIR / Healthcare Standards:** Studied FHIR R4 resource model.
  Experience with `fhir.resources` Python library (the same library
  HiperHealth uses).
- **MLOps:** MLflow for experiment tracking, ONNX for deployment-agnostic
  model export, temperature scaling for classifier calibration.
- **Testing:** pytest, coverage.py. Familiar with the 92 % coverage
  requirement in this project.
- **Tools used in this project:** ruff, mypy, pre-commit, bandit, mkdocs,
  makim (studied `.makim.yaml` to understand the build automation).

{{ Add or remove skills as appropriate for your background. }}

---

## Why Me?

I have read through the entire HiperHealth source tree — every skill, the
pipeline runner, the FHIR schema extensions, and the test suite. I understand
that the project uses an _execution-first_ philosophy: skills do one thing, do
it independently of other stages, and leave a clean audit trail. My proposal
respects that philosophy: `VisionSkill` and `NutritionSkill` each run in a
single stage, communicate only through `ctx.extras` and `ctx.results`, and
never call into each other.

I am not proposing to bolt vision onto the side of the existing system. I am
proposing to add it as a first-class, independently executable skill that
follows exactly the same conventions as `PrivacySkill`, `ExtractionSkill`,
and `DiagnosticsSkill` — so that any future contributor who has read the
existing docs will immediately understand how to extend or replace it.

---

*Proposal submitted for GSoC 2026 — Open Science Labs / HiperHealth*
*Project: MedVision + Explainability (350 hours, High complexity)*
*Mentors: Ivan Ogasawara, Aniket Kumar*
