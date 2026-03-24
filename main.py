import cv2
import numpy as np
import torch
import time

from ultralytics import YOLO
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

from dynamic_camera_test.utils.speed_estimation import compute_speed
from dynamic_camera_test.utils.direction_estimation import get_direction

#  USER INPUT 
while True:
    try:
        ppm = float(input("Enter PIXELS_PER_METER: "))
        if ppm > 0:
            break
    except:
        pass
    print("Invalid input. Try again.")

#YOLO 
model = YOLO("yolov8n.pt")

# LightGlue(optimized)
device = "cuda" if torch.cuda.is_available() else "cpu"

extractor = SuperPoint(max_num_keypoints=256).eval().to(device)  # reduced
matcher = LightGlue(features="superpoint").eval().to(device)

def frame_to_tensor(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.from_numpy(frame).float() / 255.0
    frame = frame.permute(2, 0, 1).unsqueeze(0)
    return frame.to(device)

def get_homography(prev_frame, curr_frame):
    image0 = frame_to_tensor(prev_frame)
    image1 = frame_to_tensor(curr_frame)

    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)

    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    matches = matches01["matches"]

    if len(matches) < 4:
        return np.eye(3)

    kpts0 = feats0["keypoints"][matches[:, 0]].cpu().numpy()
    kpts1 = feats1["keypoints"][matches[:, 1]].cpu().numpy()

    H, _ = cv2.findHomography(kpts0, kpts1, cv2.RANSAC)
    return H if H is not None else np.eye(3)

#VIDEO 
cap = cv2.VideoCapture("moving_cam.mp4")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, 30,
                      (int(cap.get(3)), int(cap.get(4))))

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("frame", 900, 600)

prev_frame = None
track_history = {}

frame_idx = 0
H = np.eye(3)

# FPS tracking
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    #OPTIMIZED HOMOGRAPHY 
    if prev_frame is not None and frame_idx % 5 == 0:
        small_prev = cv2.resize(prev_frame, (640, 360))
        small_curr = cv2.resize(frame, (640, 360))

        H_small = get_homography(small_prev, small_curr)

        scale_x = frame.shape[1] / 640
        scale_y = frame.shape[0] / 360

        S = np.array([[scale_x, 0, 0],
                      [0, scale_y, 0],
                      [0, 0, 1]])

        H = S @ H_small @ np.linalg.inv(S)

    # YOLO TRACKING 
    results = model.track(frame, persist=True)

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            if box.id is None:
                continue

            track_id = int(box.id[0])
            cls = int(box.cls[0])

            if cls not in [2, 3, 5, 7]:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            #track history
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append((cx, cy))

            if len(track_history[track_id]) > 10:
                track_history[track_id].pop(0)

            # speed
            speed = compute_speed(track_id, (cx, cy), H, ppm)

            #direction
            direction = get_direction(track_history[track_id])

            #label
            color = (0, 255, 0)  

            if speed == 0:
                label = f"Stopped | {direction}"
            else:
                label = f"{speed:.1f} km/h | {direction}"

        
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

            cv2.putText(frame, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, color, 2)

    # FPS CALCULATION
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # print FPS in terminal
    print(f"FPS: {fps:.2f}", end="\r")

    # show FPS on frame
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,255), 2)

    out.write(frame)

    display = cv2.resize(frame, (900, 600))
    cv2.imshow("frame", display)

    prev_frame = frame.copy()

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()