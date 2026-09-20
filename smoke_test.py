"""End-to-end smoke test: every component, in one pass, on CPU."""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

print("health  :", client.get("/health").json())

metrics = client.get("/metrics").json()
print("metrics :", {k: metrics.get(k) for k in ("available", "test_f1", "n_test")})

response = client.post(
    "/analyze", files={"file": ("demo.jpg", io.BytesIO(b"\xff\xd8\xff" + b"x" * 300), "image/jpeg")}
)
body = response.json()
print(
    f"analyze : {response.status_code} | review_required={body['decision']['review_required']}"
    f" | findings={len(body['findings'])} | reason={body['decision']['reasons'][0]}"
)

similar = client.get("/documents/cord-test-0000/similar?k=3").json()
print("similar :", [(h["doc_id"], h["similarity"]) for h in similar.get("results", [])])

rejected = client.post(
    "/analyze", files={"file": ("x.txt", io.BytesIO(b"not a document"), "text/plain")}
)
print("rejects non-document:", rejected.status_code)
