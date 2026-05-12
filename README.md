# Crowd Detection System — YOLOv8 + RTSP

Real-time person detection and crowd alerting from any RTSP camera stream.

## Features

- Detects **persons only** (COCO class 0) using YOLOv8
- Live **person count** and **FPS** overlay
- **Crowd alert** banner when count exceeds a configurable threshold
- Auto-reconnect on stream drop
- Optional **headless mode** (no display window) for server deployments
- Optional **video recording** of annotated output

---

## Installation

```bash
cd crowd_detection
pip install -r requirements.txt
```

> On first run, `ultralytics` will automatically download the chosen model weights.

---

## Usage

### Basic — display window
```bash
python detector.py --rtsp rtsp://user:pass@192.168.1.100:554/stream
```

### Choose a larger model for better accuracy
```bash
python detector.py --rtsp rtsp://... --model yolov8m.pt
```

### Adjust crowd alert threshold (default: 10 persons)
```bash
python detector.py --rtsp rtsp://... --threshold 5
```

### Headless mode + save annotated video
```bash
python detector.py --rtsp rtsp://... --no-window --save crowd_output.mp4
```

### All options
```
--rtsp        RTSP stream URL (required)
--model       YOLOv8 weights: yolov8n/s/m/l/x.pt  (default: yolov8n.pt)
--conf        Confidence threshold 0-1              (default: 0.40)
--iou         NMS IoU threshold 0-1                 (default: 0.45)
--threshold   Person count for crowd alert          (default: 10)
--no-window   Disable display window (headless)
--save FILE   Save annotated output to FILE.mp4
```

---

## Model Selection Guide

| Model        | Speed  | Accuracy | Use case                     |
|--------------|--------|----------|------------------------------|
| yolov8n.pt   | fastest | lowest  | Edge devices, low-end GPU    |
| yolov8s.pt   | fast    | good    | Balanced                     |
| yolov8m.pt   | medium  | better  | Recommended for most setups  |
| yolov8l.pt   | slow    | high    | High-accuracy deployments    |
| yolov8x.pt   | slowest | highest | Maximum accuracy             |

---

## RTSP URL Examples

| Camera brand | Typical URL format                                      |
|--------------|---------------------------------------------------------|
| Hikvision    | `rtsp://admin:pass@192.168.1.64:554/Streaming/Channels/101` |
| Dahua        | `rtsp://admin:pass@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0` |
| Generic      | `rtsp://user:pass@<ip>:<port>/stream`                   |
| Local webcam | `0`  (use device index instead of URL)                  |

---

## Project Structure

```
crowd_detection/
├── detector.py       # Main detection script
├── requirements.txt  # Python dependencies
└── README.md         # This file
```
