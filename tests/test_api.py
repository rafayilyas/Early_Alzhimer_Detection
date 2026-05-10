from __future__ import annotations

from io import BytesIO
import importlib.util
from pathlib import Path

from fastapi.testclient import TestClient


def _load_main_module():
    root = Path(__file__).resolve().parents[1]
    main_path = root / "app" / "api" / "main.py"
    spec = importlib.util.spec_from_file_location("alz_main", main_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load FastAPI module from {main_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


main = _load_main_module()


def _client():
    main.registry.load = lambda: None
    return TestClient(main.app)


def test_health_endpoint():
    client = _client()
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_mri_no_file():
    client = _client()
    response = client.post("/predict/mri")

    assert response.status_code == 422


def test_predict_mri_wrong_format():
    client = _client()
    file_obj = BytesIO(b"this is not an image")
    response = client.post(
        "/predict/mri",
        files={"file": ("sample.txt", file_obj, "text/plain")},
    )

    assert response.status_code == 400
