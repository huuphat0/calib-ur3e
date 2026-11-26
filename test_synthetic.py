#!/usr/bin/env python3
"""
Kiểm thử toàn bộ chuỗi (không cần ROS/robot): render ảnh ChArUco từ các tư thế
robot giả lập với X đã biết -> detect -> samples.json -> handeye_solve -> so X.
"""
import json, subprocess, sys, os
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from charuco_utils import (BOARD_DEFAULT, CharucoPoseEstimator, board_image, rt_to_T,
                           inv_T, rot_angle_deg)

rng = np.random.default_rng(0)
cfg = dict(BOARD_DEFAULT)
est = CharucoPoseEstimator(cfg)
W_px, H_px = 1400, 1000
flat = board_image(est.board, (W_px, H_px), 0)

# Ánh xạ pixel ảnh board -> toạ độ board (m) bằng cách detect trên ảnh phẳng
K0 = np.array([[1000, 0, W_px / 2], [0, 1000, H_px / 2], [0, 0, 1.0]])
cc, ci, _, _ = est.detect(flat, K0, np.zeros(5))
obj = est.obj_all[ci.ravel()][:, :2]
Hpix2obj, _ = cv2.findHomography(cc.reshape(-1, 2), obj)

K = np.array([[615.0, 0, 320], [0, 615.0, 240], [0, 0, 1]])   # ~RealSense 640x480
D = np.zeros(5)


def render(T_cam_board):
    R, t = T_cam_board[:3, :3], T_cam_board[:3, 3]
    H_obj2img = K @ np.column_stack([R[:, 0], R[:, 1], t])
    H = H_obj2img @ Hpix2obj
    img = cv2.warpPerspective(flat, H, (640, 480), flags=cv2.INTER_LINEAR, borderValue=200)
    img = cv2.GaussianBlur(img, (3, 3), 0.6)
    img = np.clip(img + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
    return img


def rotvec(v):
    return cv2.Rodrigues(np.asarray(v, float))[0]


# Sự thật: camera gắn cách tool0 ~ (3cm, -7cm, 4cm), xoay nhẹ
X_true = np.eye(4)
X_true[:3, :3] = rotvec([0.02, -0.03, np.pi / 2 + 0.05])
X_true[:3, 3] = [0.03, -0.07, 0.04]
# Board nằm trên bàn
W_true = np.eye(4)
# Quy ước frame board khác nhau giữa OpenCV 4.5 (y lên, z về phía người nhìn) và >=4.6
# (y xuống, z đi ra sau board). X không phụ thuộc quy ước này; chỉ cần mặt board hướng lên.
_T = np.eye(4); _T[:3, 3] = [-0.12, -0.09, 0.45]
z_away = est.estimate(render(_T), K, D)[0] is not None
W_true[:3, :3] = rotvec([0, 0, 0.3]) @ (rotvec([np.pi, 0, 0]) if z_away else np.eye(3))
W_true[:3, 3] = [0.35, 0.10, 0.0]
center_b = np.array([cfg["squares_x"] * cfg["square_len_m"] / 2, cfg["squares_y"] * cfg["square_len_m"] / 2, 0, 1])
center_w = (W_true @ center_b)[:3]

samples = []
tries = 0
while len(samples) < 20 and tries < 500:
    tries += 1
    # camera nhìn xuống board từ khoảng 35-55 cm, nghiêng ngẫu nhiên tới ~35°
    d = rng.uniform(0.35, 0.55)
    tilt = rng.uniform(0, np.radians(35)); az = rng.uniform(0, 2 * np.pi)
    view = np.array([np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), np.cos(tilt)])
    cam_pos = center_w + d * view + rng.normal(0, 0.03, 3)
    z = center_w - cam_pos; z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1.0], z); x /= np.linalg.norm(x); y = np.cross(z, x)
    R_wc = np.column_stack([x, y, z]) @ rotvec([0, 0, rng.uniform(-np.pi, np.pi)])
    T_base_cam = np.eye(4); T_base_cam[:3, :3] = R_wc; T_base_cam[:3, 3] = cam_pos
    A = T_base_cam @ inv_T(X_true)                 # T_base_tool
    T_cb_true = inv_T(T_base_cam) @ W_true
    T_cb, det = est.estimate(render(T_cb_true), K, D)
    if T_cb is None or det["n"] < 12:
        continue
    A_meas = A.copy()                              # nhiễu encoder UR ~0.1 mm
    A_meas[:3, 3] += rng.normal(0, 1e-4, 3)
    samples.append({"T_base_tool": A_meas.tolist(), "T_cam_board": T_cb.tolist(),
                    "reproj_px": det["reproj_px"], "n_corners": det["n"]})

print(f"Tạo {len(samples)} mẫu sau {tries} lần thử")

# một mẫu hỏng (board bị xê dịch 2 cm) để thử loại ngoại lai
bad = json.loads(json.dumps(samples[5]))
Tb = np.array(bad["T_cam_board"]); Tb[:3, 3] += [0.02, 0, 0]; bad["T_cam_board"] = Tb.tolist()
samples.append(bad)

T_cl_opt = np.eye(4)   # giả lập TF camera_link -> optical của RealSense
T_cl_opt[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
json.dump({"tool_frame": "tool0", "camera_link_frame": "camera_link",
           "optical_frame": "camera_color_optical_frame",
           "T_cameralink_optical": T_cl_opt.tolist(), "board": cfg, "samples": samples},
          open("samples_synth.json", "w"), indent=1)

subprocess.run([sys.executable, "handeye_solve.py", "samples_synth.json", "--out", "result_synth.json"], check=True)
X = np.array(json.load(open("result_synth.json"))["T_tool0_optical"])
et = np.linalg.norm(X[:3, 3] - X_true[:3, 3]) * 1000
er = rot_angle_deg(X[:3, :3], X_true[:3, :3])
print(f"\n=== SAI SỐ so với X thật: {et:.2f} mm, {er:.3f}° ===")
assert et < 3 and er < 0.5, "Sai số quá lớn"
print("PASS")
