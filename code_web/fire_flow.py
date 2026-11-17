import cv2
import numpy as np

# Seven rules for segmentation: https://core.ac.uk/download/pdf/55305301.pdf


def chroma_difference_mask(cb, cr, threshold=70):
    """Absolute Cb-Cr gap on signed values.

    Cb and Cr are uint8. Subtracting them in uint8 wraps, so a large gap
    such as 40 - 230 becomes 66 and fails a threshold the signed gap would pass.
    """
    return np.abs(cb.astype(np.int16) - cr.astype(np.int16)) >= threshold


def fire_pixel_segmentation(image):
    fire_mask = np.zeros_like(image[:, :, 0])

    ycrcb_im = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_ch, cr, cb = cv2.split(ycrcb_im)
    blue, green, red = cv2.split(image)

    # Rule 1: R > G > B
    mask1 = (red > green) & (green > blue)

    # Rule 2: R > 190 & G > 100 & B < 140
    mask2 = (red > 190) & (green > 100) & (blue < 140)

    # Rule 3 & 4: Y >= Cb & Cr >= Cb
    mask3_4 = (y_ch >= cb) & (cr >= cb)

    # Rule 5: Y >= Ymean & Cb <= Cbmean & Cr >= Crmean
    mask5 = (y_ch >= np.mean(y_ch)) & (cb <= np.mean(cb)) & (cr >= np.mean(cr))

    # Rule 6: |Cb - Cr| >= 70
    mask6 = chroma_difference_mask(cb, cr)

    # Rule 7: Cb <= 120 & Cr >= 150
    mask7 = (cb <= 120) & (cr >= 150)

    combined_mask = mask1 & mask2 & mask3_4 & mask5 & mask6 & mask7
    fire_mask[combined_mask] = 255

    kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(fire_mask, kernel, iterations=1)
    fire_mask = cv2.dilate(eroded_mask, kernel, iterations=1)
    return fire_mask


def push_centroid(fire_x, fire_y, centers, window):
    """Record this frame's centroid (or NaN) and return the window mean once it is full.

    Empty frames used to call np.mean([]) which warns and still stores NaN.
    The window stays one entry per frame so a quiet frame still advances it.
    """
    if centers:
        fire_x.append(float(np.mean([point[0] for point in centers])))
        fire_y.append(float(np.mean([point[1] for point in centers])))
    else:
        fire_x.append(np.nan)
        fire_y.append(np.nan)

    endpoint = None
    if len(fire_x) > window:
        xs = np.asarray(fire_x, dtype=np.float64)
        ys = np.asarray(fire_y, dtype=np.float64)
        finite = np.isfinite(xs) & np.isfinite(ys)
        if np.any(finite):
            endpoint = (int(xs[finite].mean()), int(ys[finite].mean()))
        fire_x.pop(0)
        fire_y.pop(0)
    return endpoint


def spread_direction_degrees(start_point, end_point) -> float:
    delta_x = float(end_point[0] - start_point[0])
    delta_y = float(end_point[1] - start_point[1])
    return float(np.degrees(np.arctan2(delta_y, delta_x)))


def fire_flow(fire_mask, area_frame, fire_x, fire_y, processed_frame, m):
    centers = []
    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(processed_frame, contours, -1, (0, 0, 255), 2)

    for contour in contours:
        area = cv2.contourArea(contour)
        moments = cv2.moments(contour)
        if moments["m00"] != 0 and area > 100:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            centers.append((cx, cy))

    endpoint_mframes = push_centroid(fire_x, fire_y, centers, m)

    total_area = 0
    cv2.drawContours(area_frame, contours, -1, (0, 0, 255), -1)
    binary_image = cv2.cvtColor(area_frame, cv2.COLOR_BGR2GRAY)
    _threshold, thresh = cv2.threshold(binary_image, 50, 255, cv2.THRESH_BINARY)
    area_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in area_contours:
        total_area += cv2.contourArea(contour)

    return total_area, area_frame, processed_frame, endpoint_mframes
