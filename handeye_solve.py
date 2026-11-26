#!/usr/bin/env python3
"""
Giải hand-eye EYE-IN-HAND từ samples.json  ->  T_tool0_optical (camera trên gripper).

  python3 handeye_solve.py                    # đọc samples.json
  python3 handeye_solve.py samples.json --no-reject

Phương trình:  T_base_board = T_base_tool_i · X · T_cam_board_i   (hằng số với mọi i)
Không cần ROS để chạy script này.

Đầu ra:
  handeye_result.json  (cho chế độ verify của node thu mẫu)
  handeye_tf.launch.py (static TF tool0 -> camera_link)
"""
import argparse, json, os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from charuco_utils import inv_T, mean_T, rot_angle_deg, R_to_quat, R_to_rpy

METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def solve(A, B, method):
    """A: T_base_tool list, B: T_cam_board list -> X = T_tool_cam."""
    R, t = cv2.calibrateHandEye([a[:3, :3] for a in A], [a[:3, 3] for a in A],
                                [b[:3, :3] for b in B], [b[:3, 3] for b in B], method=method)
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = R, t.ravel()
    return X


def residuals(A, B, X):
    """Độ lệch của T_base_board_i so với trung bình: (mm, deg) cho từng mẫu."""
    W = [a @ X @ b for a, b in zip(A, B)]
    Wm = mean_T(W)
    dt = np.array([np.linalg.norm(w[:3, 3] - Wm[:3, 3]) * 1000 for w in W])
    dr = np.array([rot_angle_deg(w[:3, :3], Wm[:3, :3]) for w in W])
    return dt, dr, Wm


def pose_diversity(A):
    """Góc xoay lớn nhất giữa các cặp tư thế robot + số trục xoay độc lập."""
    Rs = [a[:3, :3] for a in A]
    max_ang = max(rot_angle_deg(Rs[i], Rs[j]) for i in range(len(Rs)) for j in range(i + 1, len(Rs)))
    axes = []
    for i in range(len(Rs) - 1):
        rv = cv2.Rodrigues(Rs[i].T @ Rs[i + 1])[0].ravel()
        if np.linalg.norm(rv) > np.radians(5):
            axes.append(rv / np.linalg.norm(rv))
    rank = 0
    if axes:
        s = np.linalg.svd(np.array(axes), compute_uv=False)
        rank = int(np.sum(s > 0.3 * s[0]))
    return max_ang, rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples", nargs="?", default="samples.json")
    ap.add_argument("--no-reject", action="store_true", help="không loại mẫu ngoại lai")
    ap.add_argument("--out", default="handeye_result.json")
    a = ap.parse_args()

    data = json.load(open(a.samples))
    S = data["samples"]
    A_all = [np.array(s["T_base_tool"]) for s in S]
    B_all = [np.array(s["T_cam_board"]) for s in S]
    n = len(S)
    print(f"Số mẫu: {n}")
    if n < 8:
        sys.exit("Cần ít nhất 8 mẫu (khuyến nghị 15-25).")

    max_ang, rank = pose_diversity(A_all)
    print(f"Đa dạng tư thế: góc xoay lớn nhất giữa 2 mẫu = {max_ang:.0f}°, số trục xoay độc lập ≈ {rank}")
    if max_ang < 30 or rank < 2:
        print("  ⚠ Tư thế chưa đủ đa dạng -> kết quả có thể suy biến. Xoay cổ tay quanh ≥2 trục, 20-40°.")

    idx = list(range(n))
    # ---------- chọn method + loại ngoại lai ----------
    while True:
        A = [A_all[i] for i in idx]
        B = [B_all[i] for i in idx]
        results = {}
        for name, m in METHODS.items():
            try:
                X = solve(A, B, m)
                dt, dr, _ = residuals(A, B, X)
                results[name] = (X, dt, dr)
            except cv2.error as e:
                print(f"  {name}: lỗi {e}")
        best = min(results, key=lambda k: np.sqrt(np.mean(results[k][1] ** 2)))
        X, dt, dr = results[best]
        if a.no_reject or len(idx) <= max(10, int(0.8 * n)):
            break
        worst = int(np.argmax(dt))
        if dt[worst] > max(3 * np.median(dt), 3.0):
            print(f"  Loại mẫu #{idx[worst] + 1}: lệch {dt[worst]:.1f} mm")
            idx.pop(worst)
        else:
            break

    print("\nSo sánh phương pháp (RMS lệch T_base_board):")
    for name, (Xm, dtm, drm) in results.items():
        d = np.linalg.norm(Xm[:3, 3] - X[:3, 3]) * 1000
        print(f"  {name:10s} trans {np.sqrt(np.mean(dtm**2)):6.2f} mm | rot {np.sqrt(np.mean(drm**2)):5.2f}° "
              f"| khác best {d:5.1f} mm" + ("   <== chọn" if name == best else ""))

    spread = max(np.linalg.norm(r[0][:3, 3] - X[:3, 3]) for r in results.values()) * 1000
    rms_t, rms_r = np.sqrt(np.mean(dt ** 2)), np.sqrt(np.mean(dr ** 2))
    print(f"\nKết quả ({best}, {len(idx)}/{n} mẫu): RMS {rms_t:.2f} mm, {rms_r:.2f}°, max {dt.max():.2f} mm")
    verdict = "TỐT" if rms_t < 3 and rms_r < 0.5 else ("TẠM" if rms_t < 6 else "KÉM - nên thu lại")
    print(f"Đánh giá: {verdict}   (các phương pháp chênh nhau tối đa {spread:.1f} mm)")

    # ---------- xuất ----------
    q = R_to_quat(X[:3, :3])
    rpy = R_to_rpy(X[:3, :3])
    print("\nT_tool0_optical  (camera_color_optical_frame trong tool0):")
    print(np.array2string(X, precision=5, suppress_small=True))
    print(f"  xyz (m)   = {X[0,3]:.5f} {X[1,3]:.5f} {X[2,3]:.5f}   |t| = {np.linalg.norm(X[:3,3])*100:.1f} cm")
    print(f"  rpy (deg) = {np.degrees(rpy)[0]:.2f} {np.degrees(rpy)[1]:.2f} {np.degrees(rpy)[2]:.2f}")
    print("  -> ĐO BẰNG THƯỚC khoảng cách tâm flange tới ống kính màu, so với |t| ở trên.")

    tool = data.get("tool_frame", "tool0")
    cam_link = data.get("camera_link_frame", "camera_link")
    T_cl_opt = data.get("T_cameralink_optical")
    out = {"method": best, "n_used": len(idx), "n_total": n, "used_idx": idx,
           "rms_mm": rms_t, "rms_deg": rms_r, "T_tool0_optical": X.tolist(),
           "tool_frame": tool, "optical_frame": data.get("optical_frame")}

    if T_cl_opt is None:
        print(f"\n⚠ samples.json thiếu T_cameralink_optical (không có TF {cam_link}->optical lúc thu).")
        print("  Chỉ publish được thẳng tới optical frame -> xung đột TF với realsense driver. Xem README.")
    else:
        Y = X @ inv_T(np.array(T_cl_opt))       # T_tool0_cameralink
        qy = R_to_quat(Y[:3, :3])
        out["T_tool0_cameralink"] = Y.tolist()
        args = (f"--x {Y[0,3]:.6f} --y {Y[1,3]:.6f} --z {Y[2,3]:.6f} "
                f"--qx {qy[0]:.6f} --qy {qy[1]:.6f} --qz {qy[2]:.6f} --qw {qy[3]:.6f} "
                f"--frame-id {tool} --child-frame-id {cam_link}")
        print(f"\nStatic TF {tool} -> {cam_link}:")
        print(f"  ros2 run tf2_ros static_transform_publisher {args}")
        launch = f'''from launch import LaunchDescription
from launch_ros.actions import Node

# Sinh bởi handeye_solve.py | method={best} | RMS={rms_t:.2f} mm, {rms_r:.2f} deg | {len(idx)} mẫu
def generate_launch_description():
    return LaunchDescription([
        Node(package="tf2_ros", executable="static_transform_publisher",
             name="handeye_tool0_to_camera",
             arguments={args.split()!r}),
    ])
'''
        with open("handeye_tf.launch.py", "w") as f:
            f.write(launch)
        print("  (đã ghi handeye_tf.launch.py)")

    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Đã ghi {a.out}")


if __name__ == "__main__":
    main()
