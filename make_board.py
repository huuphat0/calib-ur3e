#!/usr/bin/env python3
"""
Sinh file in ChArUco board đúng kích thước thật (A4 ngang) + board.json.

  python3 make_board.py                      # dùng cấu hình mặc định
  python3 make_board.py --square 0.030 --marker 0.022

In ở chế độ "Actual size / 100%" (TẮT "Fit to page"). Sau khi in, đo
thước kiểm tra 100 mm trên giấy và đo cạnh 1 ô cờ bằng thước kẹp, rồi sửa
square_len_m / marker_len_m trong board.json theo số đo THẬT.
"""
import argparse, json
import cv2
from charuco_utils import BOARD_DEFAULT, make_board, board_image

MM = 72.0 / 25.4  # point per mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--square", type=float, default=BOARD_DEFAULT["square_len_m"])
    ap.add_argument("--marker", type=float, default=BOARD_DEFAULT["marker_len_m"])
    ap.add_argument("--sx", type=int, default=BOARD_DEFAULT["squares_x"])
    ap.add_argument("--sy", type=int, default=BOARD_DEFAULT["squares_y"])
    ap.add_argument("--dict", default=BOARD_DEFAULT["dictionary"])
    ap.add_argument("--out", default="charuco_board")
    a = ap.parse_args()

    cfg = {"squares_x": a.sx, "squares_y": a.sy, "square_len_m": a.square,
           "marker_len_m": a.marker, "dictionary": a.dict}
    board, _ = make_board(cfg)

    w_mm, h_mm = a.sx * a.square * 1000, a.sy * a.square * 1000
    if w_mm > 277 or h_mm > 190:
        raise SystemExit(f"Board {w_mm:.0f}x{h_mm:.0f} mm không vừa A4 ngang (lề 10 mm)")

    px_per_mm = 20  # ~508 DPI
    img = board_image(board, (int(w_mm * px_per_mm), int(h_mm * px_per_mm)), 0)
    cv2.imwrite(a.out + ".png", img)

    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.pdfgen import canvas
    pw, ph = landscape(A4)
    c = canvas.Canvas(a.out + ".pdf", pagesize=(pw, ph))
    x0 = (pw - w_mm * MM) / 2
    y0 = (ph - h_mm * MM) / 2 + 6 * MM
    c.drawImage(a.out + ".png", x0, y0, w_mm * MM, h_mm * MM)
    # thước kiểm tra 100 mm
    yb = y0 - 9 * MM
    c.setLineWidth(0.6)
    c.line(x0, yb, x0 + 100 * MM, yb)
    for i in range(11):
        hh = 3 * MM if i % 5 == 0 else 1.5 * MM
        c.line(x0 + i * 10 * MM, yb, x0 + i * 10 * MM, yb + hh)
    c.setFont("Helvetica", 7)
    c.drawString(x0 + 102 * MM, yb, "<- phai dung 100 mm")
    c.drawString(x0, 6 * MM,
                 f"ChArUco {a.sx}x{a.sy}  square={a.square*1000:.1f}mm  marker={a.marker*1000:.1f}mm  "
                 f"{a.dict}  | In 100%, KHONG 'Fit to page'. Dan len mat phang cung.")
    c.showPage()
    c.save()

    with open("board.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Đã tạo {a.out}.pdf, {a.out}.png, board.json  ({w_mm:.0f} x {h_mm:.0f} mm)")


if __name__ == "__main__":
    main()
