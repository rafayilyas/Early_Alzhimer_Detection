"""
FastAPI backend for Alzheimer's detection system.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from torchvision import models, transforms

from evaluation.gradcam import GradCAM
from evaluation.metrics import CLASS_NAMES as METRIC_CLASS_NAMES
from evaluation.metrics import generate_patient_report
from models.cnn_classifier import AlzheimerCNN
from models.dnn_tabular import AlzheimerDNN, FEATURE_NAMES


APP_VERSION = "1.1.0"
MODEL_VERSION = "1.1.0"
CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
CLASS_LABELS = CLASS_NAMES.copy()

ROOT_DIR = Path(__file__).resolve().parents[2]
CHECKPOINT_CANDIDATE_DIRS = [
    ROOT_DIR / "models" / "saved",
    ROOT_DIR / "notebooks" / "models" / "saved",
]
REPORTS_DIR = ROOT_DIR / "reports"
UPLOADS_DIR = ROOT_DIR / "data" / "uploads"
LOG_PATH    = ROOT_DIR / "predictions.log"
DB_PATH     = ROOT_DIR / "data" / "patients.db"


def _resolve_checkpoint(filename: str) -> Path:
    for directory in CHECKPOINT_CANDIDATE_DIRS:
        candidate = directory / filename
        if candidate.exists():
            return candidate
    return CHECKPOINT_CANDIDATE_DIRS[0] / filename


# Prefer the retrained checkpoint; fall back to legacy name.
CNN_CHECKPOINT    = (
    _resolve_checkpoint("cnn_best_retrained.pth")
    if _resolve_checkpoint("cnn_best_retrained.pth").exists()
    else _resolve_checkpoint("cnn_best.pth")
)
DNN_CHECKPOINT    = _resolve_checkpoint("dnn_best.pth")
FUSION_CHECKPOINT = _resolve_checkpoint("fusion_best.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

Base = declarative_base()
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class PatientRecord(Base):
    __tablename__ = "patients"

    id              = Column(Integer, primary_key=True, index=True)
    patient_id      = Column(String, unique=True, index=True, nullable=False)
    name            = Column(String, nullable=True)
    age             = Column(Integer, nullable=True)
    gender          = Column(String, nullable=True)
    diagnosis       = Column(String, nullable=True)
    mri_path        = Column(String, nullable=True)
    clinical_json   = Column(Text, nullable=True)
    prediction_json = Column(Text, nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ClinicalInput(BaseModel):
    age:                      Optional[float] = None
    gender:                   Optional[float] = None
    education_years:          Optional[float] = None
    MMSE_score:               Optional[float] = None
    CDR_score:                Optional[float] = None
    eTIV:                     Optional[float] = None
    nWBV:                     Optional[float] = None
    ASF:                      Optional[float] = None
    BMI:                      Optional[float] = None
    smoking_history:          Optional[float] = None
    family_history:           Optional[float] = None
    depression_score:         Optional[float] = None
    sleep_hours:              Optional[float] = None
    physical_activity:        Optional[float] = None
    cholesterol:              Optional[float] = None
    blood_pressure_systolic:  Optional[float] = None
    blood_pressure_diastolic: Optional[float] = None
    diabetes:                 Optional[float] = None


class PatientCreate(BaseModel):
    patient_id:    str = Field(..., min_length=1)
    name:          Optional[str] = None
    age:           Optional[int] = None
    gender:        Optional[str] = None
    diagnosis:     Optional[str] = None
    mri_path:      Optional[str] = None
    clinical_data: Optional[ClinicalInput] = None
    prediction:    Optional[Dict] = None


class PredictionResponse(BaseModel):
    predicted_class:    str
    confidence:         float
    probabilities:      Dict[str, float]
    risk_score:         float
    gradcam_image:      Optional[str] = None
    recommendation:     str
    feature_importance: Optional[Dict[str, float]] = None


class FullPredictionRequest(BaseModel):
    clinical_data: ClinicalInput


# ---------------------------------------------------------------------------
# Compatibility shim for notebook-saved checkpoints
# ---------------------------------------------------------------------------

class ClinicalTabularPreprocessor:
    pass


sys.modules["__main__"].ClinicalTabularPreprocessor = ClinicalTabularPreprocessor
if sys.modules.get("uvicorn.__main__") is not None:
    sys.modules["uvicorn.__main__"].ClinicalTabularPreprocessor = ClinicalTabularPreprocessor
sys.modules[__name__].ClinicalTabularPreprocessor = ClinicalTabularPreprocessor


# ---------------------------------------------------------------------------
# MRI model — identical twin of the training-time MRIResNet18
# (returns logits AND embedding so checkpoint keys match exactly)
# ---------------------------------------------------------------------------

class MRIResNet18(nn.Module):
    """
    ResNet-18 classifier that exposes a named layer hierarchy matching
    the training script, so torch.load() with strict=True succeeds.
    Returns (logits, embedding) — the embedding is used by the fusion model.
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        self.layer4  = backbone.layer4
        self.avgpool = backbone.avgpool
        self.embedding_dim = backbone.fc.in_features   # 512
        self.fc = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        embedding = torch.flatten(x, 1)
        logits    = self.fc(embedding)
        return logits, embedding


# ---------------------------------------------------------------------------
# Fusion model
# ---------------------------------------------------------------------------

class AlzheimerFusionModel(nn.Module):
    def __init__(self, cnn_model: MRIResNet18, dnn_model: AlzheimerDNN, num_classes: int = 4):
        super().__init__()
        self.cnn = cnn_model
        self.dnn = dnn_model
        self.cnn.eval()
        self.dnn.eval()
        for p in self.cnn.parameters():
            p.requires_grad = False
        for p in self.dnn.parameters():
            p.requires_grad = False

        self.dnn_to_cnn     = nn.Linear(128, 512)
        self.attention_gate = nn.Linear(640, 2)
        self.classifier = nn.Sequential(
            nn.Linear(640, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, mri_data: torch.Tensor, clinical_data: Optional[torch.Tensor] = None):
        if clinical_data is None:
            clinical_data = torch.zeros(
                mri_data.size(0), len(FEATURE_NAMES),
                device=mri_data.device, dtype=mri_data.dtype,
            )

        with torch.no_grad():
            _, cnn_emb = self.cnn(mri_data)
            _, dnn_emb = self.dnn(clinical_data)

        combined    = torch.cat([cnn_emb, dnn_emb], dim=1)
        attention   = torch.softmax(torch.sigmoid(self.attention_gate(combined)), dim=1)
        weighted_cnn = attention[:, 0:1] * cnn_emb
        weighted_dnn = attention[:, 1:2] * self.dnn_to_cnn(dnn_emb)
        weighted_concat = weighted_cnn + weighted_dnn
        fusion_input    = torch.cat([weighted_concat, dnn_emb], dim=1)
        logits          = self.classifier(fusion_input)
        return logits, attention


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    def __init__(self):
        self.cnn:    Optional[MRIResNet18]         = None
        self.dnn:    Optional[AlzheimerDNN]         = None
        self.fusion: Optional[AlzheimerFusionModel] = None
        self.class_names: List[str] = CLASS_NAMES.copy()
        self.loaded = False

    def load(self) -> None:
        if self.loaded:
            return

        # --- CNN ---
        self.cnn = MRIResNet18(num_classes=4).to(DEVICE)
        if CNN_CHECKPOINT.exists():
            ckpt = self._load_checkpoint(self.cnn, CNN_CHECKPOINT, strict=True)
            if isinstance(ckpt, dict):
                loaded_names = ckpt.get("class_names") or ckpt.get("classes")
                if loaded_names:
                    self.class_names = list(loaded_names)
                    logging.info("Loaded class names from checkpoint: %s", self.class_names)
        else:
            logging.warning("CNN checkpoint not found at %s — using random weights.", CNN_CHECKPOINT)

        self.cnn.eval()
        for param in self.cnn.parameters():
            param.requires_grad = False

        # --- DNN ---
        self.dnn = AlzheimerDNN(input_features=len(FEATURE_NAMES), num_classes=4).to(DEVICE)
        if DNN_CHECKPOINT.exists():
            try:
                self._load_checkpoint(self.dnn, DNN_CHECKPOINT, strict=False)
                self.dnn.eval()
                for param in self.dnn.parameters():
                    param.requires_grad = False
            except Exception as exc:
                logging.warning("Skipping clinical checkpoint — %s", exc)
                self.dnn = None
        else:
            logging.warning("DNN checkpoint not found at %s.", DNN_CHECKPOINT)
            self.dnn = None

        # --- Fusion ---
        if self.dnn is not None:
            self.fusion = AlzheimerFusionModel(self.cnn, self.dnn).to(DEVICE)
            if FUSION_CHECKPOINT.exists():
                try:
                    self._load_checkpoint(self.fusion, FUSION_CHECKPOINT, strict=False)
                    logging.info("Loaded fusion checkpoint from %s", FUSION_CHECKPOINT)
                except Exception as exc:
                    logging.warning("Skipping fusion checkpoint — %s", exc)
            self.fusion.eval()
        else:
            self.fusion = None

        self.loaded = True
        logging.info(
            "Models loaded | cnn=%s dnn=%s fusion=%s | device=%s",
            self.cnn is not None,
            self.dnn is not None,
            self.fusion is not None,
            DEVICE,
        )

    @staticmethod
    def _load_checkpoint(model: nn.Module, path: Path, *, strict: bool = False):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        if missing:
            logging.warning("Missing keys when loading %s: %s", path.name, missing)
        if unexpected:
            logging.warning("Unexpected keys when loading %s: %s", path.name, unexpected)
        return ckpt


registry = ModelRegistry()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def preprocess_mri(image_bytes: bytes) -> torch.Tensor:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.CenterCrop(160),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)


def clinical_to_tensor(clinical: ClinicalInput) -> torch.Tensor:
    values = [getattr(clinical, field) for field in FEATURE_NAMES]
    if any(v is None for v in values):
        raise HTTPException(status_code=422, detail="Missing required clinical fields")
    return torch.tensor([values], dtype=torch.float32, device=DEVICE)


def _class_names() -> List[str]:
    return registry.class_names if registry.class_names else CLASS_NAMES


def _response_dict(response: PredictionResponse) -> Dict:
    return response.model_dump() if hasattr(response, "model_dump") else response.dict()


def _risk_score_from_probabilities(probability_map: Dict[str, float]) -> float:
    severity_weights = {
        "NonDemented":       0.0,
        "VeryMildDemented": 35.0,
        "MildDemented":     70.0,
        "ModerateDemented": 100.0,
    }
    risk = sum(probability_map.get(label, 0.0) * w for label, w in severity_weights.items())
    return float(np.clip(risk, 0, 100))


def _recommendation(predicted_class: str, risk_score: float) -> str:
    if risk_score < 33:
        return "Low immediate concern. Monitor symptoms and follow up routinely."
    if risk_score < 66:
        return "Consult a neurologist within 2–4 weeks for further evaluation."
    return f"Urgent: consult a neurologist within 2 weeks. Predicted stage: {predicted_class}."


def _format_prediction(
    logits: torch.Tensor,
    gradcam_b64: Optional[str] = None,
    feature_importance: Optional[Dict[str, float]] = None,
) -> PredictionResponse:
    probabilities   = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    predicted_index = int(probabilities.argmax())
    confidence      = float(probabilities[predicted_index])
    class_names     = _class_names()
    predicted_class = class_names[predicted_index]
    probability_map = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    risk_score      = _risk_score_from_probabilities(probability_map)

    return PredictionResponse(
        predicted_class=predicted_class,
        confidence=round(confidence, 4),
        probabilities={k: round(v, 4) for k, v in probability_map.items()},
        risk_score=round(risk_score, 2),
        gradcam_image=gradcam_b64,
        recommendation=_recommendation(predicted_class, risk_score),
        feature_importance=feature_importance,
    )


def _compute_gradcam_base64(
    image_tensor: torch.Tensor,
    target_class: Optional[int] = None,
) -> str:
    if registry.cnn is None:
        raise HTTPException(status_code=503, detail="MRI model unavailable")

    cam     = GradCAM(registry.cnn, target_layer="layer4")
    heatmap = cam.generate_cam(image_tensor, target_class=target_class)
    overlay = cam.overlay_heatmap(image_tensor[0, 0].detach().cpu().numpy(), heatmap, alpha=0.4)
    cam.close()

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _local_feature_importance(model: AlzheimerDNN, clinical_tensor: torch.Tensor) -> Dict[str, float]:
    t = clinical_tensor.clone().detach().requires_grad_(True)
    logits, _ = model(t)
    target_score = logits.gather(1, logits.argmax(dim=1, keepdim=True)).sum()
    model.zero_grad(set_to_none=True)
    target_score.backward()
    importance = np.abs(t.grad.detach().cpu().numpy()[0])
    normalized = importance / importance.sum() if importance.sum() > 0 else importance
    return {FEATURE_NAMES[i]: float(round(normalized[i], 4)) for i in range(len(FEATURE_NAMES))}


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _predict_mri(image_bytes: bytes) -> PredictionResponse:
    if registry.cnn is None:
        raise HTTPException(status_code=503, detail="MRI model unavailable")

    image_tensor = preprocess_mri(image_bytes)
    with torch.no_grad():
        logits, _ = registry.cnn(image_tensor)

    gradcam_b64 = _compute_gradcam_base64(image_tensor)
    response    = _format_prediction(logits, gradcam_b64=gradcam_b64)
    logging.info("MRI prediction | %s", json.dumps(_response_dict(response)))
    return response


def _predict_clinical(clinical: ClinicalInput) -> PredictionResponse:
    if registry.dnn is None:
        raise HTTPException(status_code=503, detail="Clinical model unavailable")

    clinical_tensor = clinical_to_tensor(clinical)
    with torch.no_grad():
        logits, _ = registry.dnn(clinical_tensor)

    feature_importance = _local_feature_importance(registry.dnn, clinical_tensor)
    response = _format_prediction(logits, feature_importance=feature_importance)
    logging.info("Clinical prediction | %s", json.dumps(_response_dict(response)))
    return response


def _predict_full(image_bytes: bytes, clinical: ClinicalInput) -> PredictionResponse:
    if registry.fusion is None:
        raise HTTPException(status_code=503, detail="Fusion model unavailable")

    image_tensor    = preprocess_mri(image_bytes)
    clinical_tensor = clinical_to_tensor(clinical)

    with torch.no_grad():
        logits, _ = registry.fusion(image_tensor, clinical_tensor)

    gradcam_b64 = _compute_gradcam_base64(image_tensor)
    response    = _format_prediction(logits, gradcam_b64=gradcam_b64)
    logging.info("Fusion prediction | %s", json.dumps(_response_dict(response)))
    return response


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Alzheimer's Detection API",
    description="FastAPI backend for MRI, clinical, and fusion Alzheimer's detection",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    registry.load()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"message": "Alzheimer's Detection API", "version": APP_VERSION}


@app.get("/health")
def health():
    return {
        "status":  "ok",
        "version": APP_VERSION,
        "models": {
            "cnn":    registry.cnn    is not None,
            "dnn":    registry.dnn    is not None,
            "fusion": registry.fusion is not None,
        },
        "class_names": _class_names(),
        "device": str(DEVICE),
    }


@app.post("/predict/mri")
async def predict_mri(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Wrong file type. Please upload an image.")
    image_bytes = await file.read()
    try:
        return _response_dict(_predict_mri(image_bytes))
    except HTTPException:
        raise
    except Exception as exc:
        logging.exception("MRI prediction failure")
        raise HTTPException(status_code=500, detail=f"Model failure: {exc}") from exc


@app.post("/predict/clinical")
def predict_clinical(payload: ClinicalInput):
    try:
        return _response_dict(_predict_clinical(payload))
    except HTTPException:
        raise
    except Exception as exc:
        logging.exception("Clinical prediction failure")
        raise HTTPException(status_code=500, detail=f"Model failure: {exc}") from exc


@app.post("/predict/full")
async def predict_full(file: UploadFile = File(...), clinical_json: str = Form(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Wrong file type. Please upload an image.")

    try:
        clinical_payload = ClinicalInput(**json.loads(clinical_json))
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Missing or invalid clinical JSON") from exc

    image_bytes = await file.read()
    try:
        return _response_dict(_predict_full(image_bytes, clinical_payload))
    except HTTPException:
        raise
    except Exception as exc:
        logging.exception("Fusion prediction failure")
        raise HTTPException(status_code=500, detail=f"Model failure: {exc}") from exc


@app.post("/patients")
def save_patient_record(payload: PatientCreate, db: Session = Depends(get_db)):
    existing = db.query(PatientRecord).filter(PatientRecord.patient_id == payload.patient_id).first()
    if existing is not None:
        raise HTTPException(status_code=409, detail="Patient already exists")

    record = PatientRecord(
        patient_id      = payload.patient_id,
        name            = payload.name,
        age             = payload.age,
        gender          = payload.gender,
        diagnosis       = payload.diagnosis,
        mri_path        = payload.mri_path,
        clinical_json   = payload.clinical_data.model_dump_json() if payload.clinical_data else None,
        prediction_json = json.dumps(payload.prediction) if payload.prediction else None,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {"message": "patient saved", "patient_id": record.patient_id}


@app.get("/patients/{patient_id}")
def get_patient_history(patient_id: str, db: Session = Depends(get_db)):
    records = db.query(PatientRecord).filter(PatientRecord.patient_id == patient_id).all()
    if not records:
        raise HTTPException(status_code=404, detail="Patient not found")

    return {
        "patient_id": patient_id,
        "history": [
            {
                "name":          r.name,
                "age":           r.age,
                "gender":        r.gender,
                "diagnosis":     r.diagnosis,
                "mri_path":      r.mri_path,
                "clinical_data": json.loads(r.clinical_json)   if r.clinical_json   else None,
                "prediction":    json.loads(r.prediction_json) if r.prediction_json else None,
                "created_at":    r.created_at.isoformat()      if r.created_at      else None,
            }
            for r in records
        ],
    }


@app.get("/report/{patient_id}")
def get_report(patient_id: str, db: Session = Depends(get_db)):
    record = (
        db.query(PatientRecord)
        .filter(PatientRecord.patient_id == patient_id)
        .order_by(PatientRecord.created_at.desc())
        .first()
    )
    if record is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    if not record.mri_path or not Path(record.mri_path).exists():
        raise HTTPException(status_code=400, detail="MRI image not available for report generation")
    if not record.prediction_json:
        raise HTTPException(status_code=400, detail="Prediction data not available for report generation")

    clinical_data = json.loads(record.clinical_json) if record.clinical_json else {}
    prediction    = json.loads(record.prediction_json)

    gradcam_b64 = prediction.get("gradcam_image")
    gradcam_image = (
        Image.open(io.BytesIO(base64.b64decode(gradcam_b64)))
        if gradcam_b64
        else Image.open(record.mri_path).convert("RGB")
    )

    report_path = REPORTS_DIR / f"{patient_id}_report.pdf"
    generate_patient_report(
        patient_id    = patient_id,
        mri_path      = record.mri_path,
        clinical_data = clinical_data,
        model_outputs = prediction,
        gradcam_image = gradcam_image,
        save_path     = report_path,
    )
    return FileResponse(str(report_path), media_type="application/pdf", filename=report_path.name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)