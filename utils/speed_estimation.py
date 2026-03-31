import numpy as np
import time
from utils.motion_estimation import compensate_motion

track_history = {}
time_history = {}

def compute_speed(track_id, position, H, ppm):
    HISTORY_LENGTH = 3
    MIN_SPEED_KMPH = 0.5

    current_time = time.time()

    if track_id not in track_history:
        track_history[track_id] = []
        time_history[track_id] = []

    track_history[track_id].append(position)
    time_history[track_id].append(current_time)

    if len(track_history[track_id]) > HISTORY_LENGTH:
        track_history[track_id].pop(0)
        time_history[track_id].pop(0)

    if len(track_history[track_id]) < 2:
        return 0.0

    positions = track_history[track_id]

    start = compensate_motion(positions[0], H)
    end = positions[-1]

    dx = end[0] - start[0]
    dy = end[1] - start[1]

    pixel_dist = np.sqrt(dx**2 + dy**2)

    dt = time_history[track_id][-1] - time_history[track_id][0]
    if dt <= 0:
        return 0.0

    meter_dist = pixel_dist / ppm
    speed_kmph = (meter_dist / dt) * 3.6

    if speed_kmph < MIN_SPEED_KMPH:
        return 0.0

    return speed_kmph