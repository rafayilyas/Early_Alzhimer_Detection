# Alzheimer's Detection Project Inventory

This repository already contains the core pieces of the Alzheimer's detection system. This README documents what is present in the workspace so the project can be tracked without recreating existing assets.

## Already Present

### Data
- Dataset is present under `data/raw/MRI/Alzhiemer/combined_images/`.
- The MRI class folders are present:
  - `NonDemented`
  - `VeryMildDemented`
  - `MildDemented`
  - `ModerateDemented`
- Supporting folders exist:
  - `data/processed/`
  - `data/augmented/`
  - `data/raw/`
- The workspace also contains `data/uploads/` and a `data/patients.db` database file.

### Models
- `models/cnn_classifier.py` exists and contains the custom CNN.
- `models/pretrained_finetune.py` exists and contains pretrained architectures such as ResNet50, EfficientNet, and DenseNet.
- `models/dnn_tabular.py` exists and contains the clinical DNN.

### Training
- `training/train_cnn.py` exists.
- `training/train_dnn.py` exists.
- `training/train_fusion.py` exists.
- `training/retrain_cnn.py` exists and is used for CNN retraining.

### Evaluation
- `evaluation/metrics.py` exists.
- `evaluation/gradcam.py` exists.

### API / App
- `app/api/main.py` exists and serves the FastAPI backend.

### Notebooks
- `notebooks/01_data_exploration.ipynb` exists.
- `notebooks/02_model_training_cnn_rewritten.ipynb` exists.
- `notebooks/03_model_training_dnn.ipynb` exists.

## Not Found in This Workspace

The following items were requested, but they are not present in the current repository tree:
- `models/architectures.py`
- `evaluation/explainability.py`
- `evaluation/clinical_validation.py`
- `training/train_rl_agent.py`
- `predictions.log`
- `data/normative/`

## Notes

- The workspace currently has a root-level `README.md` documenting the verified project state.
- If you want, the next step can be to expand this README into a fuller project guide with setup, training, inference, and folder usage sections.