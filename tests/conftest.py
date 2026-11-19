import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "service"))
sys.path.insert(0, str(ROOT / "code_web"))

SAMPLES = ROOT / "service" / "public" / "samples"
