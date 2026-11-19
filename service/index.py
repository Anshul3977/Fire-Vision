"""Vercel entrypoint (zero-config FastAPI). Locally: uvicorn index:app --port 8000"""
from firevision.app import app  # noqa: F401
