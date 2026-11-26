#!/usr/bin/env python3
"""
Tiện ích ChArUco + phép biến đổi 4x4, tương thích cả 2 API của OpenCV:
  - OpenCV 4.5.x (python3-opencv của Ubuntu 22.04 / ROS 2 Humble)
  - OpenCV >= 4.7 (pip opencv-contrib-python)
Không phụ thuộc scipy.
"""
import json
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Cấu hình board mặc định (PHẢI khớp giữa lúc in và lúc detect)
#   7 x 5 ô, số hàng = 5 (LẺ) -> tránh lỗi "legacy pattern" khác nhau giữa
#   OpenCV 4.5 và >=4.6 (chỉ xảy ra với số hàng chẵn).
# ---------------------------------------------------------------------------
BOARD_DEFAULT = {
    "squares_x": 7,
    "squares_y": 5,
    "square_len_m": 0.035,   # cạnh ô cờ theo thiết kế; ĐO LẠI sau khi in!
    "marker_len_m": 0.026,   # cạnh marker ArUco trong ô
    "dictionary": "DICT_5X5_100",
}

_NEW_API = not hasattr(cv2.aruco, "CharucoBoard_create")


def load_board_cfg(path=None):
    cfg = dict(BOARD_DEFAULT)
    if path:
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg


def make_board(cfg):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg["dictionary"]))
    sx, sy = cfg["squares_x"], cfg["squares_y"]
    sq, mk = cfg["square_len_m"], cfg["marker_len_m"]
    if _NEW_API:
        board = cv2.aruco.CharucoBoard((sx, sy), sq, mk, d)
    else:
        board = cv2.aruco.CharucoBoard_create(sx, sy, sq, mk, d)
    return board, d


def board_image(board, size_px, margin_px=0):
    gen = board.generateImage if _NEW_API else board.draw
    try:
        return gen(tuple(size_px), marginSize=margin_px)
    except cv2.error:
        # Một số OpenCV lỗi làm tròn ROI với vài kích thước -> vẽ lớn hơn rồi resize
        w, h = size_px
        big = gen((w * 4 + 3, h * 4 + 3), marginSize=margin_px * 4)
        return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)


def _board_corners3d(board):
    if hasattr(board, "getChessboardCorners"):
        return np.asarray(board.getChessboardCorners(), dtype=np.float64)
    return np.asarray(board.chessboardCorners, dtype=np.float64)


class CharucoPoseEstimator:
    """Detect ChArUco và ước lượng T_cam_board (board -> camera optical frame)."""

    def __init__(self, cfg, min_corners=8):
        self.cfg = cfg
        self.board, self.dict = make_board(cfg)
        self.obj_all = _board_corners3d(self.board)
        self.min_corners = min_corners
        if _NEW_API:
            self.detector = cv2.aruco.CharucoDetector(self.board)
        else:
            self.params = cv2.aruco.DetectorParameters_create()
            self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    def detect(self, gray, K, D):
        """Trả về (ch_corners Nx1x2, ch_ids Nx1, marker_corners, marker_ids)."""
        if _NEW_API:
            ch_c, ch_i, m_c, m_i = self.detector.detectBoard(gray)
        else:
            m_c, m_i, _ = cv2.aruco.detectMarkers(gray, self.dict, parameters=self.params)
            ch_c, ch_i = None, None
            if m_i is not None and len(m_i) > 0:
                _, ch_c, ch_i = cv2.aruco.interpolateCornersCharuco(
                    m_c, m_i, gray, self.board, cameraMatrix=K, distCoeffs=D)
        return ch_c, ch_i, m_c, m_i

    def estimate(self, gray, K, D):
        """
        Trả về dict {T, n, reproj_px, ch_corners, ch_ids, m_corners, m_ids}
        hoặc None nếu không đủ góc. T = T_cam_board (4x4, mét).
        """
        ch_c, ch_i, m_c, m_i = self.detect(gray, K, D)
        out = {"ch_corners": ch_c, "ch_ids": ch_i, "m_corners": m_c, "m_ids": m_i}
        if ch_i is None or len(ch_i) < self.min_corners:
            return None, out
        ids = ch_i.ravel().astype(int)
        obj = self.obj_all[ids].reshape(-1, 1, 3)
        img = ch_c.reshape(-1, 1, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, out
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, D, rvec, tvec)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
        err = float(np.sqrt(np.mean(np.sum((proj - img) ** 2, axis=2))))
        out.update(T=rt_to_T(rvec, tvec), n=len(ids), reproj_px=err, rvec=rvec, tvec=tvec)
        return out["T"], out


# ---------------------------------------------------------------------------
# Toán 4x4 / quaternion (x, y, z, w)
# ---------------------------------------------------------------------------
def rt_to_T(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).ravel()
    return T


def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def quat_to_R(q):
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def R_to_quat(R):
    # Shepperd, ổn định số học
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q if q[3] >= 0 else -q


def T_from_tq(t, q):
    T = np.eye(4)
    T[:3, :3] = quat_to_R(q)
    T[:3, 3] = t
    return T


def R_to_rpy(R):
    """Roll-pitch-yaw (rad), quy ước ROS (R = Rz(yaw) Ry(pitch) Rx(roll))."""
    pitch = np.arcsin(-np.clip(R[2, 0], -1, 1))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def rot_angle_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def mean_rotation(Rs):
    """Trung bình rotation theo chiếu SVD (chordal L2 mean)."""
    M = np.sum(Rs, axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def mean_T(Ts):
    Ts = np.asarray(Ts)
    T = np.eye(4)
    T[:3, :3] = mean_rotation(Ts[:, :3, :3])
    T[:3, 3] = Ts[:, :3, 3].mean(axis=0)
    return T
