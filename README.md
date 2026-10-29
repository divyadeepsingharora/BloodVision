# BloodVision
**Deep learning research prototype for classifying peripheral blood-smear images as ALL or HEM.**

BloodVision explores computer-vision methods for image-based classification of acute lymphoblastic leukaemia (ALL) versus healthy/normal (HEM) peripheral blood-smear images. This repository is an educational research prototype, not a clinically validated diagnostic system.

## Current experiment
- Backbone: EfficientNet-B3 pretrained on ImageNet
- Task: binary image classification (`HEM = 0`, `ALL = 1`)
- Input size: 224 × 224
- Training framework: PyTorch / torchvision
- Dataset: CNMC 2019 / ALL-IDB-style image folders supplied separately through Kaggle
- Evaluation: patient-level train/validation/test split; report sensitivity, specificity, precision, F1, ROC-AUC, confusion matrix, and accuracy where appropriate

## Important limitations
- Image-level accuracy alone is insufficient for clinical claims.
- Split patients, not images, to reduce patient leakage.
- Results depend on dataset provenance, label quality, image acquisition, and class balance.
- Grad-CAM visualisations are exploratory and do not prove that a model uses clinically meaningful features.
- Do not use this model to diagnose, triage, or make treatment decisions. Clinical use would require independent external validation, prospective evaluation, privacy/security review, and applicable regulatory approval.

## Repository layout
```text
BloodVision/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── notebooks/
│   └── BloodVision_EfficientNetB3_Kaggle.ipynb
├── src/
│   └── train.py
├── configs/
│   └── default.json
├── docs/
│   └── DATASET.md
├── models/       # keep trained weights out of Git unless deliberately versioned
└── outputs/      # generated metrics and plots; generally not committed
```

## Run on Kaggle
1. Create a Kaggle notebook and attach the relevant blood-smear dataset as an input.
2. Import `notebooks/BloodVision_EfficientNetB3_Kaggle.ipynb` or upload it directly.
3. Check the dataset root in the notebook's **Configuration** section.
4. Enable a GPU accelerator if available.
5. Run the notebook from top to bottom.
6. Export the best checkpoint and evaluation outputs from `/kaggle/working/`.

## Run locally
Install dependencies:
```bash
pip install -r requirements.txt
```
The included Kaggle notebook is the primary end-to-end training workflow. The standalone script in `src/` is a small scaffold and expects a dataset arranged as `training/all/` and `training/hem/`.

## Dataset expectations
The notebook searches for folders named `all` and `hem`, including the previously used path:
`/kaggle/input/datasets/glendyhernandez/cnmc2019-trainprelim/new_dataset/training/`

Keep the dataset outside this repository unless its licence explicitly permits redistribution. See [docs/DATASET.md](docs/DATASET.md).

## Suggested next steps
- Verify patient identifiers and split integrity.
- Add repeated patient-level cross-validation and confidence intervals.
- Compare against a simple baseline and report calibration.
- Evaluate on a genuinely independent dataset.
- Document intended use, failure modes, and data governance before any institutional pilot.
