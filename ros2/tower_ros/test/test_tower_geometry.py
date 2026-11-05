import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower_ros.geometry import (BlockEstimate, HsvThresholds, ToolPose, color_masks, combine_samples,  # noqa: E402
                                contact_detected, estimate_block, estimate_from_marker, gripper_down_matrix,
                                linear_waypoints, marker_from_estimate, quat_gripper_down, quat_to_matrix, rgb_to_hsv,
                                table_height, tool_for_camera_above, tool_from_wrist, top_face_rect, transform_points,
                                unpack_rgb, waypoint_times, wrap_quarter, wrist_from_tool)

TABLE = -0.19


def square_face(cx, cy, side, yaw, n=600, noise=0.0, rng=None):
    rng = rng or np.random.default_rng(0)
    uv = rng.uniform(-side / 2, side / 2, (n, 2))
    c, s = math.cos(yaw), math.sin(yaw)
    xy = uv @ np.array([[c, s], [-s, c]]) + [cx, cy]
    return xy + rng.normal(0, noise, xy.shape)


def quarter_err(a, b):
    return abs(wrap_quarter(a - b))


def test_wrap_quarter():
    assert wrap_quarter(0.0) == 0.0
    assert wrap_quarter(math.radians(50)) == pytest.approx(math.radians(-40))
    assert wrap_quarter(math.radians(-100)) == pytest.approx(math.radians(-10))
    for a in np.linspace(-7, 7, 101):
        w = wrap_quarter(a)
        assert -math.pi / 4 <= w < math.pi / 4
        assert math.cos(4 * (w - a)) == pytest.approx(1.0)


def test_gripper_down_quaternion():
    assert quat_gripper_down(0.0) == pytest.approx((0.0, 1.0, 0.0, 0.0))
    for yaw in (-2.0, -0.3, 0.0, 0.7, 3.0):
        r = quat_to_matrix(quat_gripper_down(yaw))
        assert r == pytest.approx(gripper_down_matrix(yaw))
        assert r[:, 2] == pytest.approx([0, 0, -1])  # tool z points down


def test_wrist_tool_round_trip():
    for yaw in (-1.2, 0.0, 0.4):
        pose = ToolPose(0.45, -0.12, 0.03, yaw)
        xyz, q = wrist_from_tool(pose, 0.14)
        assert xyz == pytest.approx([0.45, -0.12, 0.17])
        back = tool_from_wrist(xyz, q, 0.14)
        assert back.xyz() == pytest.approx(pose.xyz())
        assert back.yaw == pytest.approx(yaw)


def test_camera_above_puts_the_camera_over_the_block():
    offset = (-0.025, 0.13, 0.0)
    for yaw in (-0.6, 0.0, 0.5):
        pose = tool_for_camera_above((0.5, 0.1), 0.25, yaw, 0.14, offset)
        wrist, q = wrist_from_tool(pose, 0.14)
        camera = wrist + quat_to_matrix(q) @ offset
        assert camera == pytest.approx([0.5, 0.1, 0.25])
        assert pose.yaw == yaw
    # yaw 0: the camera is 13 cm along +y and 2.5 cm along +x of the wrist
    pose = tool_for_camera_above((0.5, 0.1), 0.25, 0.0, 0.14, offset)
    assert (pose.x, pose.y) == pytest.approx((0.475, -0.03))


@pytest.mark.parametrize("yaw_deg", [-60, -44, -20, 0, 7, 33, 45, 80])
def test_top_face_rect_numpy(yaw_deg):
    yaw = math.radians(yaw_deg)
    xy = square_face(0.42, -0.05, 0.04, yaw, noise=0.0005)
    (cx, cy), (w, l), est = top_face_rect(xy, use_cv2=False)
    assert quarter_err(est, yaw) < math.radians(1.5)
    assert -math.pi / 4 <= est < math.pi / 4
    assert math.hypot(cx - 0.42, cy + 0.05) < 1.5e-3
    assert w == pytest.approx(0.04, abs=3e-3) and l == pytest.approx(0.04, abs=3e-3)


@pytest.mark.parametrize("yaw_deg", [-60, -20, 0, 7, 33, 80])
def test_top_face_rect_cv2_matches(yaw_deg):
    pytest.importorskip("cv2")
    yaw = math.radians(yaw_deg)
    xy = square_face(0.3, 0.2, 0.05, yaw, noise=0.0005)
    (cx, cy), _, est = top_face_rect(xy, use_cv2=True)
    assert quarter_err(est, yaw) < math.radians(1.5)
    assert math.hypot(cx - 0.3, cy - 0.2) < 1.5e-3


def test_rgb_to_hsv_opencv_convention():
    rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 0, 0], [128, 128, 128], [255, 255, 0]], np.uint8)
    hsv = rgb_to_hsv(rgb)
    assert hsv.tolist() == [[0, 255, 255], [60, 255, 255], [120, 255, 255], [0, 0, 0], [0, 0, 128], [30, 255, 255]]


def test_rgb_to_hsv_matches_cv2():
    cv2 = pytest.importorskip("cv2")
    rgb = np.random.default_rng(1).integers(0, 256, (2000, 3), dtype=np.uint8)
    ref = cv2.cvtColor(rgb[:, None, ::-1].copy(), cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)
    ours = rgb_to_hsv(rgb).astype(int)
    dh = np.abs(ours[:, 0] - ref[:, 0])
    assert np.minimum(dh, 180 - dh).max() <= 1
    assert np.abs(ours[:, 1:] - ref[:, 1:]).max() <= 1


def test_unpack_rgb():
    rgb = np.array([[10, 200, 30], [255, 0, 128]], np.uint32)
    packed = ((rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]).astype(np.uint32).view(np.float32)
    assert unpack_rgb(packed).tolist() == rgb.tolist()


def test_transform_points():
    yaw = 0.3
    q = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    out = transform_points(np.array([[1.0, 0.0, 0.0]]), (0.1, 0.2, 0.3), q)
    assert out[0] == pytest.approx([0.1 + math.cos(yaw), 0.2 + math.sin(yaw), 0.3])


def scene(rng, blocks):
    """Table points plus box tops and sides; blocks: (color hsv, cx, cy, side, height, yaw)."""
    tx, ty = np.meshgrid(np.arange(0.2, 0.8, 0.004), np.arange(-0.3, 0.3, 0.004))
    pts = [np.stack([tx.ravel(), ty.ravel(), TABLE + rng.normal(0, 0.001, tx.size)], 1)]
    cols = [np.tile([20, 30, 160], (tx.size, 1))]  # light table
    for hsv, cx, cy, side, height, yaw in blocks:
        inside = np.zeros(len(pts[0]), bool)
        c, s = math.cos(yaw), math.sin(yaw)
        d = pts[0][:, :2] - [cx, cy]
        u, v = d @ [c, s], d @ [-s, c]
        inside |= (np.abs(u) < side / 2 + 0.002) & (np.abs(v) < side / 2 + 0.002)
        pts[0], cols[0] = pts[0][~inside], cols[0][~inside]
        top = square_face(cx, cy, side, yaw, n=500, rng=rng)
        pts.append(np.column_stack([top, np.full(len(top), TABLE + height) + rng.normal(0, 0.001, len(top))]))
        cols.append(np.tile(hsv, (len(top), 1)))
        side_pts = square_face(cx, cy, side, yaw, n=300, rng=rng)
        pts.append(np.column_stack([side_pts, TABLE + rng.uniform(0, height, len(side_pts))]))
        cols.append(np.tile(hsv, (len(side_pts), 1)))
    return np.concatenate(pts), np.concatenate(cols).astype(np.uint8)


def test_segmentation_and_block_estimate():
    rng = np.random.default_rng(3)
    green = ([60, 200, 150], 0.45, -0.10, 0.05, 0.05, math.radians(12))
    black = ([0, 0, 30], 0.55, 0.12, 0.04, 0.05, math.radians(-30))
    red = ([2, 220, 180], 0.35, 0.08, 0.035, 0.04, math.radians(40))
    points, hsv = scene(rng, [green, black, red])
    tz = table_height(points[:, 2], -0.4, 0.1)
    assert tz == pytest.approx(TABLE, abs=2e-3)
    masks = color_masks(hsv, points[:, 2], tz, HsvThresholds())
    any_block = masks["green"] | masks["black"] | masks["red"]
    assert not (masks["green"] & masks["black"]).any()
    for name, (_, cx, cy, side, height, yaw) in zip(("green", "black", "red"), (green, black, red)):
        e = estimate_block(points, masks[name], any_block, tz)
        assert e is not None, name
        assert math.hypot(e.x - cx, e.y - cy) < 2e-3, name
        assert quarter_err(e.yaw, yaw) < math.radians(2), name
        assert e.height == pytest.approx(height, abs=2e-3), name
        assert e.top == pytest.approx(TABLE + height, abs=2e-3), name
        assert e.bottom == pytest.approx(TABLE, abs=2e-3), name
    # the black top offset raises the top and the height, not the bottom
    e = estimate_block(points, masks["black"], any_block, tz, top_offset=0.02)
    assert e.height == pytest.approx(0.07, abs=2e-3) and e.bottom == pytest.approx(TABLE, abs=2e-3)


def test_dark_table_shadow_is_not_black_block():
    hsv = np.array([[0, 0, 20], [0, 0, 20]], np.uint8)
    z = np.array([TABLE + 0.001, TABLE + 0.03])
    assert color_masks(hsv, z, TABLE, HsvThresholds())["black"].tolist() == [False, True]


def test_combine_samples_wraps_yaw():
    a = BlockEstimate(0.1, 0.2, 0.05, math.radians(44), 0.05, 100)
    b = BlockEstimate(0.1, 0.2, 0.05, math.radians(-44), 0.05, 100)
    yaw = combine_samples([a, b, a, b]).yaw
    assert quarter_err(yaw, math.radians(45)) < 1e-9
    c = combine_samples([BlockEstimate(0.1, 0.2, z, 0.1, 0.05) for z in (0.050, 0.051, 0.090)])
    assert c.top == pytest.approx(0.051)  # median drops the outlier


def test_marker_round_trip():
    e = BlockEstimate(0.4, -0.1, -0.09, math.radians(20), 0.1, 321)
    center, size_z = marker_from_estimate(e)
    yaw = e.yaw
    back = estimate_from_marker(center, (0, 0, math.sin(yaw / 2), math.cos(yaw / 2)), size_z, 321)
    assert (back.x, back.y, back.top, back.height) == pytest.approx((e.x, e.y, e.top, e.height))
    assert back.yaw == pytest.approx(e.yaw)


def test_contact_detected():
    base = np.array([1.0, -2.0, 5.0])
    quiet = base + np.random.default_rng(0).normal(0, 0.5, (10, 3))
    assert not contact_detected(quiet, base, 3.0)
    assert contact_detected(quiet + [0, 0, 4.0], base, 3.0)
    spike = quiet.copy()
    spike[-1] += [0, 0, 20.0]  # one bad sample out of ten does not stop the arm
    assert not contact_detected(spike, base, 3.0)
    assert not contact_detected(np.zeros((0, 3)), base, 3.0)


def test_linear_waypoints_and_times():
    a, b = ToolPose(0.4, 0.0, 0.10, 0.0), ToolPose(0.4, 0.0, 0.088, 0.0)
    wps = linear_waypoints(a, b, 0.001)
    assert len(wps) == 12
    assert wps[-1].xyz() == pytest.approx(b.xyz())
    steps = np.diff([a.z] + [w.z for w in wps])
    assert np.allclose(steps, -0.001)
    times = waypoint_times(a, wps, 0.01, min_dt=0.0)
    assert times[-1] == pytest.approx(1.2)  # 12 mm at 1 cm/s
    assert np.all(np.diff(times) > 0)
    # yaw goes the short way round and limits the step count
    turn = linear_waypoints(ToolPose(0, 0, 0, 3.0), ToolPose(0, 0, 0, -3.0), 0.005)
    assert abs(turn[-1].yaw - 3.0) == pytest.approx(2 * math.pi - 6.0)
