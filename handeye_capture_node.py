#!/usr/bin/env python3
"""
Node ROS 2 thu mẫu hand-eye (EYE-IN-HAND: camera gắn trên gripper UR3).

Chạy (trong container Humble, đã chạy ur_robot_driver + realsense2_camera):
  python3 handeye_capture_node.py
  python3 handeye_capture_node.py --ros-args -p image_topic:=/camera/camera/color/image_raw

Phím (cửa sổ ảnh phải đang được focus):
  s : chụp 1 mẫu (tự lấy trung bình ~10 frame, từ chối nếu robot/ảnh rung)
  u : xoá mẫu cuối
  q : lưu và thoát
Mỗi mẫu được tự lưu ngay vào samples.json.

Chế độ kiểm chứng (sau khi đã giải):
  python3 handeye_capture_node.py --ros-args -p result:=handeye_result.json
  -> hiển thị pose board trong base_link; di chuyển robot, số phải gần như KHÔNG ĐỔI.

Node này KHÔNG điều khiển robot, chỉ đọc TF.
"""
import json, os, sys, threading, time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from tf2_ros import Buffer, TransformListener

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from charuco_utils import (CharucoPoseEstimator, load_board_cfg, T_from_tq, inv_T,
                           mean_T, rot_angle_deg)

N_AVG = 10               # số frame lấy trung bình mỗi mẫu
MAX_TF_JITTER_MM = 0.5   # robot phải đứng yên
MAX_CAM_JITTER_MM = 2.0  # pose board ổn định
MAX_REPROJ_PX = 1.0


def img_msg_to_bgr(msg):
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    enc = msg.encoding.lower()
    if enc in ("rgb8", "bgr8"):
        img = buf.reshape(msg.height, msg.step)[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if enc == "rgb8" else img.copy()
    if enc in ("mono8", "8uc1"):
        g = buf.reshape(msg.height, msg.step)[:, :msg.width]
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    if enc in ("rgba8", "bgra8"):
        img = buf.reshape(msg.height, msg.step)[:, :msg.width * 4].reshape(msg.height, msg.width, 4)
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if enc == "rgba8" else cv2.COLOR_BGRA2BGR)
    if enc in ("yuv422_yuy2", "yuyv"):
        img = buf.reshape(msg.height, msg.width, 2)
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUYV)
    raise ValueError(f"Encoding chưa hỗ trợ: {msg.encoding} (đặt rgb_camera.color_format:=RGB8)")


def tf_to_T(tf):
    t, q = tf.transform.translation, tf.transform.rotation
    return T_from_tq([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])


class HandEyeCapture(Node):
    def __init__(self):
        super().__init__("handeye_capture")
        p = self.declare_parameter
        self.base = p("base_frame", "base_link").value
        self.tool = p("tool_frame", "tool0").value
        self.cam_link = p("camera_link_frame", "camera_link").value
        self.image_topic = p("image_topic", "").value
        self.info_topic = p("info_topic", "").value
        self.out_path = p("out", "samples.json").value
        board_path = p("board", "board.json").value
        result_path = p("result", "").value

        if not os.path.exists(board_path):
            self.get_logger().warn(f"Không thấy {board_path}, dùng cấu hình board mặc định")
            board_path = None
        self.cfg = load_board_cfg(board_path)
        self.est = CharucoPoseEstimator(self.cfg)
        self.get_logger().info(f"Board: {self.cfg}")

        self.X = None
        if result_path:
            with open(result_path) as f:
                self.X = np.array(json.load(f)["T_tool0_optical"])
            self.get_logger().info("CHẾ ĐỘ KIỂM CHỨNG: dùng X từ " + result_path)

        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)
        self.lock = threading.Lock()
        self.frame, self.K, self.D, self.optical_frame = None, None, None, None

        self._resolve_topics()
        self.create_subscription(Image, self.image_topic, self._on_img, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, self.info_topic, self._on_info, qos_profile_sensor_data)

    # ---------------- topics ----------------
    def _resolve_topics(self):
        if not self.image_topic:
            cands = []
            for _ in range(20):  # đợi discovery tối đa ~4 s
                topics = self.get_topic_names_and_types()
                cands = [n for n, ts in topics
                         if "sensor_msgs/msg/Image" in ts and n.endswith("color/image_raw")]
                if cands:
                    break
                time.sleep(0.2)
            if len(cands) != 1:
                imgs = [n for n, ts in self.get_topic_names_and_types() if "sensor_msgs/msg/Image" in ts]
                self.get_logger().error(
                    f"Không tự xác định được topic ảnh màu. Tìm thấy: {cands or imgs or 'KHÔNG CÓ topic Image nào'}\n"
                    "  -> kiểm tra realsense2_camera đã chạy chưa (ros2 topic list | grep color),\n"
                    "     rồi chỉ định: --ros-args -p image_topic:=<topic>")
                raise SystemExit(1)
            self.image_topic = cands[0]
        if not self.info_topic:
            self.info_topic = self.image_topic.rsplit("/", 1)[0] + "/camera_info"
        self.get_logger().info(f"Ảnh: {self.image_topic} | Intrinsic: {self.info_topic}")

    def _on_info(self, msg):
        with self.lock:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.D = np.array(msg.d, dtype=np.float64) if len(msg.d) else np.zeros(5)
            self.optical_frame = msg.header.frame_id
            self.img_size = (msg.width, msg.height)

    def _on_img(self, msg):
        try:
            img = img_msg_to_bgr(msg)
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=5.0)
            return
        with self.lock:
            self.frame = img

    # ---------------- helpers ----------------
    def lookup(self, parent, child):
        try:
            tf = self.tf_buf.lookup_transform(parent, child, rclpy.time.Time())
            return tf_to_T(tf)
        except Exception as e:  # noqa
            self.get_logger().warn(f"TF {parent}->{child}: {e}", throttle_duration_sec=2.0)
            return None

    def snapshot(self):
        with self.lock:
            if self.frame is None or self.K is None:
                return None, None, None
            return self.frame.copy(), self.K.copy(), self.D.copy()


def draw_text(img, lines, color=(0, 255, 0)):
    y = 25
    for ln in lines:
        cv2.putText(img, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        y += 24


def main():
    rclpy.init()
    node = HandEyeCapture()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()

    samples = []
    if os.path.exists(node.out_path) and node.X is None:
        with open(node.out_path) as f:
            samples = json.load(f).get("samples", [])
        node.get_logger().info(f"Nạp {len(samples)} mẫu cũ từ {node.out_path} (xoá file để làm lại từ đầu)")

    def save():
        T_cl_opt = node.lookup(node.cam_link, node.optical_frame) if node.optical_frame else None
        data = {"base_frame": node.base, "tool_frame": node.tool,
                "optical_frame": node.optical_frame, "camera_link_frame": node.cam_link,
                "T_cameralink_optical": T_cl_opt.tolist() if T_cl_opt is not None else None,
                "K": node.K.tolist() if node.K is not None else None,
                "D": node.D.tolist() if node.D is not None else None,
                "board": node.cfg, "samples": samples}
        with open(node.out_path, "w") as f:
            json.dump(data, f, indent=1)

    win = "handeye_capture (s=chup, u=xoa, q=thoat)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    collecting, buf_tool, buf_cam, msg, msg_color, msg_until = False, [], [], "", (0, 255, 0), 0
    verify_ref = None

    while rclpy.ok():
        img, K, D = node.snapshot()
        if img is None:
            blank = np.zeros((240, 640, 3), np.uint8)
            draw_text(blank, ["Dang doi anh + camera_info ...", node.image_topic], (0, 200, 255))
            cv2.imshow(win, blank)
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        T_cb, det = node.est.estimate(gray, K, D)
        vis = img.copy()
        if det["m_ids"] is not None and len(det["m_ids"]):
            cv2.aruco.drawDetectedMarkers(vis, det["m_corners"], det["m_ids"])
        if det["ch_ids"] is not None and len(det["ch_ids"]):
            cv2.aruco.drawDetectedCornersCharuco(vis, det["ch_corners"], det["ch_ids"], (0, 0, 255))
        T_bt = node.lookup(node.base, node.tool)

        lines = [f"Mau: {len(samples)}  (nen 15-25)"]
        good = T_cb is not None and det["reproj_px"] < MAX_REPROJ_PX
        if T_cb is not None:
            cv2.drawFrameAxes(vis, K, D, det["rvec"], det["tvec"], 0.05)
            dist = np.linalg.norm(T_cb[:3, 3])
            lines.append(f"Goc: {det['n']}  reproj: {det['reproj_px']:.2f}px  khoang cach: {dist*100:.0f}cm")
        else:
            lines.append("Khong thay board (can >= 8 goc ChArUco)")
        if T_bt is None:
            lines.append(f"CHUA CO TF {node.base}->{node.tool}")

        # gợi ý đa dạng tư thế
        if T_bt is not None and samples:
            dmin = min(rot_angle_deg(np.array(s["T_base_tool"])[:3, :3], T_bt[:3, :3]) for s in samples)
            lines.append(f"Goc xoay khac mau gan nhat: {dmin:.0f} deg" + ("  (xoay them!)" if dmin < 10 else ""))

        # chế độ kiểm chứng
        if node.X is not None and T_cb is not None and T_bt is not None:
            T_base_board = T_bt @ node.X @ T_cb
            if verify_ref is None:
                verify_ref = T_base_board
            dt = np.linalg.norm(T_base_board[:3, 3] - verify_ref[:3, 3]) * 1000
            dr = rot_angle_deg(T_base_board[:3, :3], verify_ref[:3, :3])
            p = T_base_board[:3, 3] * 1000
            lines.append(f"[VERIFY] board trong {node.base}: x={p[0]:.1f} y={p[1]:.1f} z={p[2]:.1f} mm")
            lines.append(f"[VERIFY] lech so voi luc dau: {dt:.1f} mm, {dr:.2f} deg (r = reset)")

        # thu trung bình
        if collecting:
            if good and T_bt is not None:
                buf_tool.append(T_bt)
                buf_cam.append(T_cb)
            lines.append(f"Dang thu {len(buf_cam)}/{N_AVG} ... GIU YEN ROBOT")
            if len(buf_cam) >= N_AVG:
                collecting = False
                bt, bc = np.array(buf_tool), np.array(buf_cam)
                jt = np.max(np.linalg.norm(bt[:, :3, 3] - bt[:, :3, 3].mean(0), axis=1)) * 1000
                jc = np.max(np.linalg.norm(bc[:, :3, 3] - bc[:, :3, 3].mean(0), axis=1)) * 1000
                if jt > MAX_TF_JITTER_MM or jc > MAX_CAM_JITTER_MM:
                    msg, msg_color = f"TU CHOI: rung robot {jt:.2f}mm / board {jc:.2f}mm. Thu lai.", (0, 0, 255)
                else:
                    samples.append({"T_base_tool": mean_T(bt).tolist(),
                                    "T_cam_board": mean_T(bc).tolist(),
                                    "reproj_px": float(det["reproj_px"]) if T_cb is not None else None,
                                    "n_corners": int(det["n"]) if T_cb is not None else None,
                                    "stamp": time.time()})
                    save()
                    msg, msg_color = f"Da luu mau #{len(samples)}", (0, 255, 0)
                msg_until = time.time() + 2.5
            elif time.time() > collect_deadline:
                collecting = False
                msg, msg_color, msg_until = "Het thoi gian: board mat/reproj cao. Thu lai.", (0, 0, 255), time.time() + 2.5

        if time.time() < msg_until:
            lines.append(msg)
        draw_text(vis, lines, msg_color if time.time() < msg_until else (0, 255, 0))
        cv2.imshow(win, vis)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("s") and not collecting and node.X is None:
            if not good or T_bt is None:
                msg, msg_color, msg_until = "Khong chup duoc: can thay board ro + co TF", (0, 0, 255), time.time() + 2
            else:
                collecting, buf_tool, buf_cam = True, [], []
                collect_deadline = time.time() + 5.0
        elif k == ord("u") and samples:
            samples.pop()
            save()
            msg, msg_color, msg_until = f"Da xoa, con {len(samples)} mau", (0, 200, 255), time.time() + 2
        elif k == ord("r"):
            verify_ref = None

    if node.X is None:
        save()
        print(f"Đã lưu {len(samples)} mẫu vào {node.out_path}")
    cv2.destroyAllWindows()
    ex.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
