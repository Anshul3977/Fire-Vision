import cv2
import numpy as np


def smoke_pixel_segmentation(image):
    hsv_im = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 50, 0], dtype=np.uint8)
    upper_bound = np.array([180, 200, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv_im, lower_bound, upper_bound)
    invert_mask = cv2.bitwise_not(mask)

    kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(invert_mask, kernel, iterations=1)
    smoke_mask = cv2.dilate(eroded_mask, kernel, iterations=1)
    return smoke_mask


def mean_flow_angle(flow_x, flow_y):
    """Degrees of the mean motion vector. None when there is nothing to average."""
    if len(flow_x) == 0 or len(flow_y) == 0:
        return None
    average_x = float(np.mean(flow_x))
    average_y = float(np.mean(flow_y))
    if not np.isfinite(average_x) or not np.isfinite(average_y):
        return None
    if average_x == 0.0 and average_y == 0.0:
        return None
    return float(np.degrees(np.arctan2(average_y, average_x)))


def push_direction(history, angle, window):
    """Append this frame's angle (NaN if unknown) and return the window mean when full."""
    history.append(np.nan if angle is None else float(angle))
    if len(history) <= window:
        return None
    values = np.asarray(history, dtype=np.float64)
    finite = values[np.isfinite(values)]
    history.pop(0)
    if finite.size == 0:
        return None
    return float(finite.mean())


def smoke_flow(im, prev, smoke_mask, wind_dir, processed_frame, n):
    curr = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # pyr_scale, levels, winsize, iterations, poly_n, poly_sigma
    flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 5, 3, 1.5, 0)

    flow_x = []
    flow_y = []
    step = 15
    for y in range(0, processed_frame.shape[0], step):
        for x in range(0, processed_frame.shape[1], step):
            if smoke_mask[y, x] > 0:
                fx, fy = flow[y, x]
                flow_magnitude = np.sqrt(fx**2 + fy**2)
                if flow_magnitude > 0.5:
                    flow_x.append(fx)
                    flow_y.append(fy)
                    end_point = (int(x + fx), int(y + fy))
                    cv2.arrowedLine(processed_frame, (x, y), end_point, (255, 0, 0), 2, tipLength=3)

    angle_in_degrees = mean_flow_angle(flow_x, flow_y)
    if angle_in_degrees is None:
        label = "Smoke Direction: n/a"
    else:
        label = f"Smoke Direction: {angle_in_degrees:.2f}"
    cv2.putText(
        processed_frame,
        label,
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    smoke_dir_nframes = push_direction(wind_dir, angle_in_degrees, n)
    return curr, processed_frame, smoke_dir_nframes
