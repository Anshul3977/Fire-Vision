# Forest Fire Detection System

A real-time fire and smoke detection system using computer vision and deep learning techniques. The system analyzes video feeds to detect, track, and analyze fire spread patterns and smoke movement.

## Features

- **Fire Detection**: YOLOv8-based object detection for fire and smoke
- **Fire Spread Analysis**: Tracks fire progression and calculates spread direction using optical flow
- **Smoke Detection**: Detects smoke patterns and movement direction
- **Real-time Monitoring**: Live video processing with WebSocket streaming
- **Alert System**: SMS and call alerts via Twilio integration
- **Analytics Dashboard**: Visual graphs showing burned area, spread rate, and smoke dispersion

## Tech Stack

- **Backend**: Flask, Flask-SocketIO
- **Computer Vision**: OpenCV, YOLOv8 (Ultralytics)
- **Analysis**: NumPy, Matplotlib
- **Alerts**: Twilio API
- **Frontend**: HTML, CSS, JavaScript, jQuery

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt --break-system-packages
```

3. Set up environment variables:
```bash
export TWILIO_ACCOUNT_SID="your_account_sid"
export TWILIO_AUTH_TOKEN="your_auth_token"
export TWILIO_PHONE="your_twilio_phone"
export ALERT_PHONE="recipient_phone"
```

## Usage

1. Start the Flask server:
```bash
cd code_web
python app.py
```

2. Open browser and navigate to `http://localhost:5000`

3. Upload a video file and configure analysis parameters:
   - **Size Factor**: Resize video for faster processing (1-2 recommended)
   - **Frames for Smoke Analysis**: Number of frames to average smoke direction (default: 30)
   - **Frames for Fire Analysis**: Number of frames to track fire spread (default: 180)
   - **Camera Stable**: Enable for fixed camera footage to analyze fire/smoke movement

4. Click "Upload Video" to start processing

## Key Components

### Fire Detection (`fire_flow.py`)
- 7-rule pixel segmentation for fire detection
- Contour-based fire area calculation
- Fire centroid tracking for spread analysis

### Smoke Detection (`smoke_flow.py`)
- HSV color space thresholding
- Optical flow analysis using Farneback method
- Smoke direction calculation

### YOLO Detection (`yolo_detection.py`)
- Pre-trained YOLOv8 model for fire/smoke detection
- Oriented bounding box support
- Real-time object tracking

### Analysis (`analysis.py`)
- 4-panel visualization dashboard:
  - Cumulative burned area over time
  - Rate of area change
  - Smoke dispersion direction (polar plot)
  - Fire spread path with ignition and current front markers

## Alert System

The system automatically sends alerts when fire or smoke is detected:
- **SMS Alert**: Text message via Twilio
- **Voice Call**: Automated call notification
- Alerts are triggered once per session to prevent spam

## Model Performance

| Model | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|-----------|--------|--------|-----------|
| YOLO-v8-obb PT v0 | 92.5% | 89.2% | 94.5% | 63.2% |
| YOLO-v8 PT | 90.0% | 83.0% | 89.0% | 55.3% |
| YOLO-v5-NU | 89.9% | 81.1% | 87.2% | 55.2% |

## Project Structure

```
├── code_web/
│   ├── app.py              # Main Flask application
│   ├── fire_flow.py        # Fire detection logic
│   ├── smoke_flow.py       # Smoke detection logic
│   ├── yolo_detection.py   # YOLO model inference
│   ├── analysis.py         # Graph generation
│   ├── static/             # CSS, JS, JSON files
│   └── templates/          # HTML templates
├── notebooks/              # Jupyter notebooks for experimentation
├── models/                 # Pre-trained YOLO models
└── requirements.txt        # Python dependencies
```

## Contributing

Contributions are welcome! This project was developed for forest fire management research.
