"""One-off helper: render a side-by-side qualitative comparison (GT / baseline /
mod_1 / mod_2) of projected 3D boxes for a chosen validation image, for use in
the report's qualitative-results section."""
import os
import sys
import numpy as np
import cv2

ROOT = "/root/workdir/diploma/MonoDTR"
DATA = os.path.join(ROOT, "data/KITTI/object/training")
SCORE_TH = 0.5


def panels_for(img_id):
    return [
        ("Ground Truth", os.path.join(DATA, "label_2", img_id + ".txt"), False, (0, 200, 0)),
        ("Baseline (MonoDTR)", os.path.join(ROOT, "workdirs/MonoDTR/output/validation/data", img_id + ".txt"), True, (60, 60, 230)),
        ("Modification 1", os.path.join(ROOT, "workdirs/MonoDTR_mod_1/output/validation/data", img_id + ".txt"), True, (230, 60, 60)),
        ("Modification 2", os.path.join(ROOT, "workdirs/MonoDTR_mod_2/output/validation/data", img_id + ".txt"), True, (40, 170, 230)),
    ]


def read_objects(path, is_pred):
    objs = []
    with open(path) as f:
        for line in f:
            p = line.strip().split(" ")
            if p[0] != "Car":
                continue
            score = float(p[15]) if is_pred and len(p) > 15 else None
            if is_pred and score < SCORE_TH:
                continue
            h, w, l = (float(p[8]), float(p[9]), float(p[10]))
            x, y, z = (float(p[11]), float(p[12]), float(p[13]))
            ry = float(p[14])
            objs.append({"hwl": (h, w, l), "xyz": (x, y, z), "ry": ry, "score": score})
    return objs


def read_p2(calib_path):
    with open(calib_path) as f:
        for line in f:
            if line.startswith("P2:"):
                return np.array([float(x) for x in line.strip().split(" ")[1:]]).reshape(3, 4)
    raise RuntimeError("P2 not found")


def box_3d_corners(h, w, l, x, y, z, ry):
    # KITTI convention: (x, y, z) is the bottom-center of the box in camera coords
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners = np.array([x_corners, y_corners, z_corners])
    R = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)],
    ])
    corners = R @ corners
    corners[0, :] += x
    corners[1, :] += y
    corners[2, :] += z
    return corners  # [3, 8]


EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
         (4, 5), (5, 6), (6, 7), (7, 4),  # top face
         (0, 4), (1, 5), (2, 6), (3, 7)]  # verticals
FRONT_EDGES = {(0, 1), (1, 5), (5, 4), (4, 0)}  # highlight the "front" face of the car


def draw_box(img, corners_3d, p2, color):
    pts = p2 @ np.vstack([corners_3d, np.ones((1, 8))])
    if np.any(pts[2] <= 0.1):
        return
    pts_2d = (pts[:2] / pts[2]).T  # [8, 2]
    pts_2d = pts_2d.astype(int)
    for a, b in EDGES:
        thickness = 3 if (a, b) in FRONT_EDGES or (b, a) in FRONT_EDGES else 1
        cv2.line(img, tuple(pts_2d[a]), tuple(pts_2d[b]), color, thickness, cv2.LINE_AA)


def main(img_id):
    img_path = os.path.join(DATA, "image_2", img_id + ".png")
    p2 = read_p2(os.path.join(DATA, "calib", img_id + ".txt"))
    base_img = cv2.imread(img_path)
    h, w = base_img.shape[:2]

    panels = []
    for title, label_path, is_pred, color in panels_for(img_id):
        canvas = base_img.copy()
        objs = read_objects(label_path, is_pred)
        for o in objs:
            corners = box_3d_corners(*o["hwl"], *o["xyz"], o["ry"])
            draw_box(canvas, corners, p2, color)
        text = f"{title}  ({len(objs)} boxes" + (f", score>={SCORE_TH}" if is_pred else "") + ")"
        (tw, th_), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(canvas, (0, 0), (tw + 16, th_ + 18), (255, 255, 255), -1)
        cv2.putText(canvas, text, (8, th_ + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        panels.append(canvas)

    pad = 8
    grid = np.full((h * 2 + pad, w * 2 + pad, 3), 255, dtype=np.uint8)
    grid[0:h, 0:w] = panels[0]
    grid[0:h, w + pad:] = panels[1]
    grid[h + pad:, 0:w] = panels[2]
    grid[h + pad:, w + pad:] = panels[3]

    out_path = os.path.join(ROOT, f"resources/qual_example_{img_id}.png")
    cv2.imwrite(out_path, grid)
    print("wrote", out_path, grid.shape)


if __name__ == "__main__":
    main(sys.argv[1])
