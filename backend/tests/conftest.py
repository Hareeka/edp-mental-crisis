import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ.setdefault("MC_DATABASE_URL", f"sqlite:///{_tmp}/test.db")
os.environ.setdefault("MC_NLP_BACKEND", "lexicon")
os.environ.setdefault("MC_RISK_MODEL_PATH", os.path.join(_tmp, "missing.joblib"))
os.environ.setdefault("MC_ENCRYPTION_KEY", "Zm9vYmFyYmF6cXV4Zm9vYmFyYmF6cXV4Zm9vYmFyYmE=")
os.environ.setdefault("MC_JWT_SECRET", "test-secret")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth(client):
    import uuid

    r = client.post("/api/auth/register", json={"alias": "t", "email": f"{uuid.uuid4().hex}@ex.com",
                                                "password": "password123"})
    assert r.status_code == 201
    return {"Authorization": f"Bearer {r.json()['access_token']}"}
