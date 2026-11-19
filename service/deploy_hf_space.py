"""Publish the in-browser demo (hf-space/) to a free Hugging Face *static* Space.

    HF_TOKEN=hf_xxx python deploy_hf_space.py --space <hf-username>/fire-vision-demo

(Docker/Gradio Spaces on free CPU now need HF PRO, so the FastAPI service runs on Vercel.)
"""
import argparse, os, shutil, tempfile
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--space", required=True)
a = ap.parse_args()
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(a.space, repo_type="space", space_sdk="static", exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    stage = Path(tmp)
    shutil.copytree(HERE / "hf-space", stage, dirs_exist_ok=True)
    shutil.copy(HERE / "models" / "fire_smoke_dfire.onnx", stage)
    for f in ("flame-front.jpg", "forest-burn.jpg", "forest-burn-pan.mp4"):
        shutil.copy(HERE / "public" / "samples" / f, stage)
    api.upload_folder(folder_path=stage, repo_id=a.space, repo_type="space", commit_message="Deploy from GitHub")
print(f"https://huggingface.co/spaces/{a.space}")
