FOCAL_LENGTH = 800
REAL_CAR_HEIGHT = 1.5

def estimate_distance(pixel_height):
    if pixel_height == 0:
        return 0
    return (REAL_CAR_HEIGHT * FOCAL_LENGTH) / pixel_height