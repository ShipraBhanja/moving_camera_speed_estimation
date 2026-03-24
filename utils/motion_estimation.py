import numpy as np

def compensate_motion(point, H):
    px = np.array([point[0], point[1], 1])
    transformed = H @ px
    transformed /= transformed[2]
    return int(transformed[0]), int(transformed[1])