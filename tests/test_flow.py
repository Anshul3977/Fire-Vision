import warnings

import cv2
import numpy as np

from analysis import graph, ordered_unique
from fire_flow import (
    chroma_difference_mask,
    fire_pixel_segmentation,
    push_centroid,
    spread_direction_degrees,
)
from smoke_flow import mean_flow_angle, push_direction, smoke_flow


def test_rule6_uses_signed_difference_where_uint8_wraps():
    cb = np.array([[40]], dtype=np.uint8)
    cr = np.array([[230]], dtype=np.uint8)
    wrapped = int(np.abs(cb - cr)[0, 0])
    assert wrapped < 70
    assert bool(chroma_difference_mask(cb, cr)[0, 0])
    # The other direction does not wrap the same way.
    assert int(np.abs(np.array([230], np.uint8) - np.array([40], np.uint8))[0]) >= 70


def test_fire_segmentation_accepts_a_flame_block_and_rejects_blue():
    flame = np.zeros((48, 48, 3), dtype=np.uint8)
    flame[:] = (40, 120, 255)
    blue = np.zeros_like(flame)
    blue[:] = (255, 0, 0)
    assert fire_pixel_segmentation(flame)[24, 24] == 255
    assert fire_pixel_segmentation(blue)[24, 24] == 0


def test_empty_centroid_does_not_warn_and_window_uses_finite_points():
    fire_x, fire_y = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert push_centroid(fire_x, fire_y, [], 2) is None
        assert push_centroid(fire_x, fire_y, [(10, 10)], 2) is None
        endpoint = push_centroid(fire_x, fire_y, [(30, 30)], 2)
    assert endpoint == (20, 20)


def test_spread_direction_follows_image_axes():
    assert spread_direction_degrees((0, 0), (10, 0)) == pytest_approx(0)
    assert spread_direction_degrees((0, 0), (0, 10)) == pytest_approx(90)
    assert spread_direction_degrees((5, 5), (5, 0)) == pytest_approx(-90)


def pytest_approx(value):
    import pytest

    return pytest.approx(value)


def test_mean_flow_angle_and_empty_vectors():
    assert mean_flow_angle([], []) is None
    assert mean_flow_angle([2.0], [0.0]) == pytest_approx(0)
    assert mean_flow_angle([0.0], [2.0]) == pytest_approx(90)
    assert mean_flow_angle([0.0], [0.0]) is None


def test_smoke_window_ignores_missing_angles_without_warning():
    empty = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert push_direction(empty, None, 1) is None
        assert push_direction(empty, None, 1) is None
        history = []
        assert push_direction(history, 0.0, 1) is None
        assert push_direction(history, None, 1) == pytest_approx(0)


def test_farneback_reports_a_rightward_shift():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (90, 120), dtype=np.uint8)
    previous = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    shifted = np.zeros_like(previous)
    shifted[:, 6:] = previous[:, :-6]
    mask = np.full(base.shape, 255, dtype=np.uint8)
    prev_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    _curr, _frame, _summary = smoke_flow(shifted, prev_gray, mask, [], shifted.copy(), 30)
    # The immediate angle is drawn from this pair; read it back from a one-step window.
    history = []
    angle = None
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY), None, 0.5, 3, 15, 5, 3, 1.5, 0
    )
    xs, ys = [], []
    for y in range(0, flow.shape[0], 15):
        for x in range(0, flow.shape[1], 15):
            fx, fy = flow[y, x]
            if np.hypot(fx, fy) > 0.5:
                xs.append(fx)
                ys.append(fy)
    angle = mean_flow_angle(xs, ys)
    assert angle is not None
    assert abs(angle) < 25
    assert push_direction(history, angle, 0) == pytest_approx(angle)


def test_path_order_and_graph_png():
    assert ordered_unique([(2, 2), (1, 1), (2, 2)]) == [(2, 2), (1, 1)]
    encoded = graph(
        [0, 1, 2, 3], [0, 1, 2, 4], [10.0, float("nan"), -20.0], [(2, 2), (1, 1), (2, 2)], dpi=50
    )
    assert encoded
    raw = __import__("base64").b64decode(encoded)
    assert raw.startswith(b"\x89PNG")
    assert graph([0, 1], [0, 0], [], [], dpi=40).startswith("iVBOR")
