import numpy as np
from utils.motion import compensate_motion

prev_positions = {}

def compute_speed(track_id, position, H, ppm):
    if track_id in prev_positions:
        prev_pos = prev_positions[track_id]

        prev_comp = compensate_motion(prev_pos, H)

        dx = position[0] - prev_comp[0]
        dy = position[1] - prev_comp[1]

        pixel_dist = np.sqrt(dx**2 + dy**2)

        FPS = 30

        # 🔥 use user input ppm
        speed = (pixel_dist / ppm) * FPS
    else:
        speed = 0

    prev_positions[track_id] = position
    return speed