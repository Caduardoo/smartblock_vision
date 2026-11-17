# SmartBlock Vision

Real-time color and intensity classification (Red, Green, Blue; Light, Medium, Dark) using HSV segmentation and OpenCV, supporting DVR-IP CCTV cameras, RTSP streams, and generic USB webcams.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(Optional) To use the direct DVR-IP protocol `(option 1)`, clone python-dvr inside the project root:

```bash
git clone https://github.com/OpenIPC/python-dvr
```

## Usage

```bash
    python smartblock.py
```

Press `q` (or `ESC`) on the video window to stop streaming and optionally export the annotated video and final frame.
