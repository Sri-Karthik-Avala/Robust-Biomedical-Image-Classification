# Robust Biomedical Image Classification

| | |
| --- | --- |
| Final rank | 3rd |
| Domain | Computer Vision |
| Difficulty | Medium |
| Scoring | ↑ Higher is better |
| Compute | A10G |
| Challenge status | Accepted / closed |
| Solutions submitted | 2 |
| Last submission | 2026-07-01 |

## Problem statement

### BioShift-108: Biomedical Recognition Under Unseen Acquisition Shifts

### Overview

In this challenge, participants predict anonymized biomedical image labels from compact NumPy image arrays.

The data spans multiple biomedical imaging groups, including:

- Microscopy
- X-ray
- Retinal imaging
- Ultrasound
- CT
- 3D biomedical volumes converted into 2D RGB projections

The task is **multi-class image classification**. Each target is an anonymized class label such as:

```
class_000
class_001
class_002
```

A public `classes.csv` file lists every valid prediction label. The prepared task contains **108 anonymized biomedical classes**.

This challenge is designed around **robust biomedical recognition under hidden acquisition shifts**. Training samples contain generic visual corruptions such as blur, brightness, contrast, and noise. The private test set contains stronger, modality-specific biomedical acquisition shifts that are not present during training, including stain changes in microscopy, acoustic artifacts in ultrasound, field-of-view artifacts in retinal imaging, windowing shifts in CT, detector artifacts in X-ray, and projection changes for 3D-derived samples.

Strong solutions should learn biomedical visual patterns that remain stable across unseen acquisition conditions. Models that perform well only on easy modalities or familiar corruption types will be penalized by the evaluation metric.

---

### Dataset

The dataset is organized as follows:

```
public/
├── train/
│   └── images.npz
├── test/
│   ├── images.npz
│   └── metadata.csv
├── train.csv
├── classes.csv
└── sample_submission.csv

private/
└── answers.csv
```

Both `public/train/images.npz` and `public/test/images.npz` contain one array named:

```
images
```

Each array has compact RGB images of shape:

```
(N, 64, 64, 3)
```

Images are stored as `uint8` arrays.

---

### Files and Columns

`public/train.csv`

| Column | Type | Description |
| --- | --- | --- |
| `image_id` | string | Training sample identifier |
| `filename` | string | Path to the NPZ file containing the image array |
| `npz_index` | integer | Position of the image inside the NPZ array |
| `domain_id` | string | Anonymized biomedical source-domain identifier |
| `modality_group` | string | Broad biomedical imaging group |
| `corruption_type` | string | Visual shift applied to the image |
| `corruption_family` | string | Either `generic` or `bio_specific` |
| `corruption_severity` | integer | Shift severity |
| `is_unseen_corruption` | boolean | Whether this corruption family is held out from training |
| `label` | string | Ground-truth anonymized biomedical class label |

Training images only contain generic corruptions. Therefore, `is_unseen_corruption` is `False` for all training rows.

`public/test/metadata.csv`

| Column | Type | Description |
| --- | --- | --- |
| `image_id` | string | Test sample identifier |
| `filename` | string | Path to the NPZ file containing the image array |
| `npz_index` | integer | Position of the image inside the NPZ array |
| `domain_id` | string | Anonymized biomedical source-domain identifier |
| `modality_group` | string | Broad biomedical imaging group |
| `corruption_type` | string | Public corruption descriptor. Hidden biomedical shifts may be shown as `hidden_biomedical_shift` |
| `corruption_family` | string | Either `generic` or `bio_specific` |
| `corruption_severity` | integer | Shift severity |
| `is_unseen_corruption` | boolean | Whether this corruption family is held out from training |

The exact private biomedical corruption type is not necessarily revealed in the public test metadata. This prevents participants from hard-coding shift-specific rules based only on metadata.

`public/classes.csv`

The file `public/classes.csv` contains every valid prediction label.

| Column | Type | Description |
| --- | --- | --- |
| `class_id` | integer | Integer index of the anonymized class |
| `label` | string | Valid prediction label |

`private/answers.csv`

This file is used only by the evaluator.

| Column | Type | Description |
| --- | --- | --- |
| `image_id` | string | Test sample identifier |
| `label` | string | Ground-truth anonymized biomedical class label |
| `modality_group` | string | Broad biomedical imaging group |
| `corruption_type` | string | Exact private corruption type used for robustness scoring |
| `corruption_family` | string | Either `generic` or `bio_specific` |
| `corruption_severity` | integer | Shift severity |
| `is_unseen_corruption` | boolean | Whether this corruption family is held out from training |

---

### Submission

Submit a CSV file with exactly two columns:

```
image_id,prediction
test_000000,class_000
test_000001,class_001
test_000002,class_002
```

Predictions must use labels from:

```
public/classes.csv
```

Each test `image_id` must appear exactly once.

Submissions may be rejected if they contain:

- Duplicate IDs
- Missing predictions
- Invalid labels
- Extra columns

---

### Evaluation

Submissions are evaluated using a robustness-weighted score:

```
Final Score = 0.50 * OverallMacroF1
            + 0.25 * WorstModalityMacroF1
            + 0.25 * WorstCorruptionMacroF1
```

### Overall Macro-F1

`OverallMacroF1` is the standard Macro-F1 across all labels present in the private test set.

For each class:

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

Then:

```
OverallMacroF1 = mean(F1 for every private test class)
```

### Worst-Modality Macro-F1

`WorstModalityMacroF1` is the minimum Macro-F1 over biomedical modality groups, such as microscopy, X-ray, retinal imaging, ultrasound, CT, and 3D-derived projections.

This penalizes solutions that work well on one biomedical domain but fail on another.

### Worst-Corruption Macro-F1

`WorstCorruptionMacroF1` is the minimum Macro-F1 over private corruption types.

This penalizes solutions that perform well on common or easy shifts but collapse under a specific unseen biomedical acquisition shift.

---

### What Not To Use

The following are **not allowed**:

- External biomedical image datasets
- Medical-domain pretrained models
- Models pretrained or fine-tuned on the source data used to build this challenge
- Test-time lookup, nearest-neighbor matching, or hash matching against any public source collection
- Reconstruction of hidden labels from source files, original sample order, or source-specific artifacts
- Use of private labels or `private/answers.csv` during training

General-purpose ImageNet-pretrained backbones are allowed.

However, models pretrained on biomedical datasets or task-specific medical images are not allowed.

---

### Originality Note

Unlike a standard compact biomedical classification benchmark, this task evaluates whether models can generalize to **unseen, modality-specific acquisition shifts**. The private test set combines cross-modality classification, anonymized labels, hidden biomedical corruptions, and worst-group robustness scoring.

A strong submission must be class-discriminative, modality-robust, and resilient to private corruption types that are absent from training.
