import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code_web"))


@pytest.fixture
def client():
    from app import create_app

    app = create_app(testing=True)
    return app.test_client()
