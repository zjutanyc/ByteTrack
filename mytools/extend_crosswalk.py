"""
读取四点 JSON（包含 image_width/height 与 points），绘制四个点并保存。
输出 JSON 默认命名为 extend_<原名>.json，图像保存为 extend_<原名>.png，均在输入 JSON 所在目录。
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Sequence

import cv2
import numpy as np
from dataclasses import dataclass

POINT_COLOR = (0, 0, 255)
LINE_COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class Edge:
    p1: np.ndarray
    p2: np.ndarray
    length: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load crosswalk JSON, draw points on a blank canvas, and save."
    )
    parser.add_argument(
        "-j", "--json", required=True, help="Path to input JSON with points and image size."
    )
    parser.add_argument(
        "--ratio",
        default="0.5,0.5",
        help="Comma-separated ratios for extending the two short edges (shortest, other). Default 0.5,0.5",
    )
    return parser.parse_args()


def load_json(path: Path) -> Tuple[int, int, List[Tuple[int, int]]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    w = int(data["image_width"])
    h = int(data["image_height"])
    pts = [(int(p["x"]), int(p["y"])) for p in data["points"]]
    if len(pts) != 4:
        raise ValueError(f"Expected 4 points, got {len(pts)}")
    return w, h, pts


def draw_points(width: int, height: int, pts: List[Tuple[int, int]], base_image: np.ndarray = None) -> np.ndarray:
    if base_image is not None and base_image.shape[0] == height and base_image.shape[1] == width:
        canvas = base_image.copy()
    else:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
    poly = np.array(pts, dtype=np.int32)
    cv2.polylines(canvas, [poly], isClosed=True, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    for idx, (x, y) in enumerate(pts):
        cv2.circle(canvas, (x, y), 5, POINT_COLOR, -1)
        cv2.putText(
            canvas, str(idx + 1), (x + 6, y - 6), FONT, 0.6, POINT_COLOR, 2, cv2.LINE_AA
        )
    return canvas


def save_outputs(input_json: Path, image: np.ndarray, points: List[Tuple[int, int]], width: int, height: int) -> None:
    out_dir = input_json.parent
    stem = input_json.stem  # e.g., zebra_points
    img_path = out_dir / f"extend_{stem}.png"
    cv2.imwrite(str(img_path), image)
    print(f"Saved image to {img_path}")

    json_out = out_dir / f"extend_{stem}.json"
    data = {
        "image_width": int(width),
        "image_height": int(height),
        "points": [{"x": int(x), "y": int(y)} for x, y in points],
    }
    with json_out.open("w", encoding="utf-8") as f_out:
        json.dump(data, f_out, indent=2)
    print(f"Saved JSON to {json_out}")

def build_edges(points: List[Tuple[int, int]]) -> List[Edge]:
    """
    L1: p0-p1, L2: p1-p2, L3: p2-p3, L4: p3-p0
    """
    pts = np.array(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Points must be length 4")
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    edges: List[Edge] = []
    for i1, i2 in pairs:
        p1, p2 = pts[i1], pts[i2]
        edges.append(Edge(p1=p1, p2=p2, length=float(np.linalg.norm(p2 - p1))))
    return edges


def extend_short_edges(points: List[Tuple[int, int]], ratio: float) -> List[Tuple[int, int]]:
    """
    根据对边长度确定短边组（L1+L3 vs L2+L4），对短边组按 (1+ratio) 等比例延长，返回新四点。
    """
    if isinstance(ratio, (list, tuple, Sequence)):
        if len(ratio) != 2:
            raise ValueError("ratio must have 2 elements when using a sequence")
        r_shortest, r_other = float(ratio[0]), float(ratio[1])
    else:
        r_shortest = r_other = float(ratio)

    pts = np.array(points, dtype=np.float32)
    edges = build_edges(points)
    sum_pair1 = edges[0].length + edges[2].length  # L1 + L3
    sum_pair2 = edges[1].length + edges[3].length  # L2 + L4

    if sum_pair1 <= sum_pair2:
        short_pairs = [(0, 1, edges[0].length), (2, 3, edges[2].length)]
    else:
        short_pairs = [(1, 2, edges[1].length), (3, 0, edges[3].length)]

    # 按长度排序，短的用 r_shortest，长的用 r_other
    short_pairs_sorted = sorted(short_pairs, key=lambda x: x[2])
    ratios_to_use = [r_shortest, r_other]
    for (i1, i2, _len_edge), r_use in zip(short_pairs_sorted, ratios_to_use):
        v = pts[i2] - pts[i1]
        extend_vec = v * (r_use / 2.0)
        pts[i1] = pts[i1] - extend_vec
        pts[i2] = pts[i2] + extend_vec

    new_pts = [(int(round(x)), int(round(y))) for x, y in pts]
    return new_pts


def line_to_border(p1: np.ndarray, p2: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    将通过 p1,p2 的直线截取到图像边界，返回两个边界交点（float）。
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    eps = 1e-8
    candidates = []

    # x = 0
    if abs(dx) > eps:
        t = -x1 / dx
        y = y1 + t * dy
        if 0 <= y <= height - 1:
            candidates.append((t, np.array([0.0, y])))
    # x = width-1
    if abs(dx) > eps:
        t = (width - 1 - x1) / dx
        y = y1 + t * dy
        if 0 <= y <= height - 1:
            candidates.append((t, np.array([width - 1.0, y])))
    # y = 0
    if abs(dy) > eps:
        t = -y1 / dy
        x = x1 + t * dx
        if 0 <= x <= width - 1:
            candidates.append((t, np.array([x, 0.0])))
    # y = height-1
    if abs(dy) > eps:
        t = (height - 1 - y1) / dy
        x = x1 + t * dx
        if 0 <= x <= width - 1:
            candidates.append((t, np.array([x, height - 1.0])))

    if len(candidates) < 2:
        # 退化情况：点重合或线平行且在界外
        return p1, p2

    # 按 t 排序，取最小和最大
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1], candidates[-1][1]


def extend_long_edges(points: List[Tuple[int, int]], width: int, height: int) -> List[Tuple[int, int]]:
    """
    找出长边组（L1+L3 vs L2+L4 中较长的一组），将这两条长边延长到画面边缘，返回更新后的四个点。
    """
    pts = np.array(points, dtype=np.float32)
    edges = build_edges(points)
    sum_pair1 = edges[0].length + edges[2].length  # L1 + L3
    sum_pair2 = edges[1].length + edges[3].length  # L2 + L4

    if sum_pair1 >= sum_pair2:
        long_pairs = [(0, 1), (2, 3)]
    else:
        long_pairs = [(1, 2), (3, 0)]

    new_pts = pts.copy()
    for i1, i2 in long_pairs:
        p_ext1, p_ext2 = line_to_border(pts[i1], pts[i2], width, height)
        # 为保持拓扑，将更靠近原 p_i1 的点放在 i1
        if np.linalg.norm(p_ext1 - pts[i1]) <= np.linalg.norm(p_ext2 - pts[i1]):
            new_pts[i1] = p_ext1
            new_pts[i2] = p_ext2
        else:
            new_pts[i1] = p_ext2
            new_pts[i2] = p_ext1

    return [(int(round(x)), int(round(y))) for x, y in new_pts]


def reorder_to_max_quad(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    确保返回的四点按凸四边形顺序排列，避免自交，取最大凸包。
    """
    pts = np.array(points, dtype=np.float32)
    if pts.shape[0] != 4:
        raise ValueError("Need exactly 4 points to reorder")

    hull = cv2.convexHull(pts, returnPoints=True)
    hull_pts = hull[:, 0, :] if hull.ndim == 3 else hull
    if hull_pts.shape[0] < 4:
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        order = np.argsort(angles)
        ordered = pts[order]
    else:
        ordered = hull_pts

    if ordered.shape[0] != 4:
        # 冗余安全处理
        center = ordered.mean(axis=0)
        angles = np.arctan2(ordered[:, 1] - center[1], ordered[:, 0] - center[0])
        order = np.argsort(angles)
        ordered = ordered[order]

    area = 0.5 * np.sum(
        ordered[:, 0] * np.roll(ordered[:, 1], -1)
        - ordered[:, 1] * np.roll(ordered[:, 0], -1)
    )
    if area < 0:
        ordered = ordered[::-1]

    return [(int(round(x)), int(round(y))) for x, y in ordered]


def main():
    args = parse_args()
    json_path = Path(args.json)
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    ratio_strs = [s.strip() for s in args.ratio.split(",") if s.strip()]
    if len(ratio_strs) == 1:
        ratios = (float(ratio_strs[0]), float(ratio_strs[0]))
    elif len(ratio_strs) == 2:
        ratios = (float(ratio_strs[0]), float(ratio_strs[1]))
    else:
        raise ValueError("ratio must be one or two comma-separated numbers")

    w, h, pts = load_json(json_path)
    extend_pts = extend_short_edges(pts, ratio=ratios)
    extend_pts = extend_long_edges(extend_pts, w, h)
    extend_pts = reorder_to_max_quad(extend_pts)
    base_img = None
    candidate_img = json_path.with_suffix(".png")
    if candidate_img.is_file():
        img = cv2.imread(str(candidate_img))
        if img is not None and img.shape[0] == h and img.shape[1] == w:
            base_img = img
    canvas = draw_points(w, h, extend_pts, base_image=base_img)
    save_outputs(json_path, canvas, extend_pts, w, h)


if __name__ == "__main__":
    main()
