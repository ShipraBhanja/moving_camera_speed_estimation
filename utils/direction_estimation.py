def get_direction(track):
    if len(track) < 2:
        return "Unknown"

    dx = track[-1][0] - track[0][0]
    return "Right" if dx > 0 else "Left"