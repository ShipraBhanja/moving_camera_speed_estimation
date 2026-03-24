# Vehicle Speed Estimation (Moving Camera)

This project estimates vehicle speed from video captured by a **moving camera** using computer vision.

---

## Tech Stack

* YOLOv8 (object detection + tracking)
* LightGlue + SuperPoint (feature matching)
* OpenCV (processing + visualization)
* NumPy (math operations)

---

## Features

* Vehicle detection and tracking
* Camera motion compensation using homography
* Speed estimation in **km/h**
* Handles stationary vehicles
* Real-time FPS display
* Saves output video

---

##  How to Run

### 1. Install dependencies

```bash
pip install opencv-python numpy torch torchvision ultralytics
pip install git+https://github.com/cvg/LightGlue.git
```

---

### 2. Run the script

```bash
python main.py
```

---

### 3. Input required

You will be prompted for:

```
PIXELS_PER_METER
```

---

## Output

* Bounding boxes with:

  * Speed (km/h)
  * Direction
* Output video saved as:

```
output.mp4
```

---

## Notes

* Accuracy depends on correct pixel-to-meter ratio
* Works best for road scenes with minimal elevation change
* LightGlue improves accuracy but reduces speed

---
