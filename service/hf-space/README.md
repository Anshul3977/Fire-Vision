---
title: Fire-Vision (in-browser)
emoji: 🔥
colorFrom: red
colorTo: gray
sdk: static
pinned: false
license: agpl-3.0
short_description: Fire + smoke detection with ONNX Runtime Web
---

# Fire-Vision, in your browser

A YOLOv8n fire + smoke detector, fine-tuned on the D-Fire dataset (test mAP50 0.611), running **client-side** with ONNX Runtime Web. Upload a photo or a short clip. Clips are sampled at 4 frames/s, and an alert would fire after 3 consecutive positive frames. Nothing leaves your device.

- Server version (FastAPI + ONNX Runtime, `/api/detect` and `/api/detect/video`): https://fire-vision-anshul.vercel.app
- Source, benchmarks, tests and training script: https://github.com/Anshul3977/Fire-Vision

Sample photos: U.S. Fish & Wildlife Service, public domain. Training data: D-Fire (Venâncio et al., 2022, CC BY 4.0).
