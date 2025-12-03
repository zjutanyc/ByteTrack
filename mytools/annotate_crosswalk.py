"""
- 在单张图片上用鼠标点击斑马线四个角点，实时显示编号与连线。
- 保存时会输出 JSON（点坐标）与带标注的 PNG，命名为 <原图名>_points.json / .png。
- 可选用 -o 指定输出目录；未指定则保存在原图所在目录。
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


POINT_COLOR = (0, 255, 0)
LINE_COLOR = (0, 165, 255)
TEXT_COLOR = (0, 0, 0)
INSTRUCTION_COLOR = (0, 200, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Click four points for the zebra crossing corners"
    )
    parser.add_argument(
        "-i",
        "--image",
        required=True,
        help="Path to the image where you want to annotate the crossing.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Directory to save results. Default: same folder as the image.",
    )
    parser.add_argument(
        "--window",
        default="crossing-annotator",
        help="Window name used by OpenCV.",
    )
    return parser.parse_args()


class PointCollector:
    def __init__(self, image: np.ndarray, window_name: str) -> None:
        self._image = image
        self.window_name = window_name
        self.points: List[Tuple[int, int]] = []
        self._preview = self._image.copy()
        h, w = image.shape[:2]
        self.center_x = w / 2.0
        self.center_y = h / 2.0
        self.zoom = 1.0
        self._view_info = (0, 0, 1.0)  # x0, y0, scale

    def reset(self) -> None:
        self.points = []
        self._preview = self._image.copy()
        h, w = self._image.shape[:2]
        self.center_x = w / 2.0
        self.center_y = h / 2.0
        self.zoom = 1.0

    def add_point(self, x: int, y: int) -> None:
        if len(self.points) >= 4:
            return
        self.points.append((x, y))
        self._draw_preview()

    def undo(self) -> None:
        if not self.points:
            return
        self.points.pop()
        self._draw_preview()

    def _draw_preview(self) -> None:
        canvas = self._image.copy()
        if len(self.points) >= 2:
            closed = len(self.points) == 4
            pts = np.array(self.points, dtype=np.int32)
            cv2.polylines(canvas, [pts], closed, LINE_COLOR, 2, cv2.LINE_AA)

        for idx, (x, y) in enumerate(self.points):
            cv2.circle(canvas, (x, y), 5, POINT_COLOR, -1)
            cv2.putText(
                canvas,
                str(idx + 1),
                (x + 6, y - 6),
                FONT,
                0.6,
                TEXT_COLOR,
                2,
                cv2.LINE_AA,
            )

        self._preview = canvas

    def render_with_instructions(self) -> np.ndarray:
        canvas, _ = self._compute_view()
        instructions = [
            "Left click: add point (max 4)",
            "Arrows pan   +/- zoom   u undo   r reset",
            "s save    q/ESC quit",
        ]
        y_base = 30
        for idx, text in enumerate(instructions):
            cv2.putText(
                canvas,
                text,
                (15, y_base + idx * 24),
                FONT,
                0.9,
                INSTRUCTION_COLOR,
                2,
                cv2.LINE_AA,
            )
        return canvas

    def annotated_image(self) -> np.ndarray:
        return self._preview.copy()

    def zoom_in(self) -> None:
        self.zoom = min(self.zoom * 1.1, 10.0)

    def zoom_out(self) -> None:
        self.zoom = max(self.zoom / 1.1, 1.0)

    def pan(self, dx_sign: int, dy_sign: int) -> None:
        step = max(10.0, 50.0 / self.zoom)
        self.center_x += dx_sign * step
        self.center_y += dy_sign * step

    def _compute_view(self) -> Tuple[np.ndarray, Tuple[int, int, float]]:
        h, w = self._preview.shape[:2]
        crop_w = max(50, int(w / self.zoom))
        crop_h = max(50, int(h / self.zoom))
        half_w = crop_w // 2
        half_h = crop_h // 2
        cx = np.clip(self.center_x, half_w, w - half_w)
        cy = np.clip(self.center_y, half_h, h - half_h)
        x0 = int(cx - half_w)
        y0 = int(cy - half_h)
        x1 = x0 + crop_w
        y1 = y0 + crop_h
        crop = self._preview[y0:y1, x0:x1]
        scale = w / float(crop_w)
        view = cv2.resize(crop, (w, h))
        self._view_info = (x0, y0, scale)
        return view, self._view_info

    def render_view(self) -> np.ndarray:
        view, _ = self._compute_view()
        return view

    def map_display_to_image(self, x: int, y: int) -> Tuple[int, int]:
        x0, y0, scale = self._view_info
        real_x = int(x / scale + x0)
        real_y = int(y / scale + y0)
        return real_x, real_y


def save_points(points: List[Tuple[int, int]], output_path: Path, image_shape: Tuple[int, int]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = image_shape
    data = {
        "image_width": int(width),
        "image_height": int(height),
        "points": [{"x": int(x), "y": int(y)} for x, y in points],
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(points)} points to {output_path}")


def save_annotated_image(image: np.ndarray, output_path: Path) -> None:
    cv2.imwrite(str(output_path), image)
    print(f"Saved annotated image to {output_path}")


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 决定输出目录与统一命名：<原图名>_points.{json|png}
    if args.output:
        output_dir = Path(args.output)
        if output_dir.suffix:  # 如果用户传了文件名，取其所在目录
            output_dir = output_dir.parent
    else:
        output_dir = image_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{image_path.stem}_points"
    json_path = output_dir / f"{base_name}.json"
    annotated_path = output_dir / f"{base_name}.png"

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    collector = PointCollector(image, args.window)
    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)

    def mouse_handler(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            real_x, real_y = collector.map_display_to_image(x, y)
            collector.add_point(real_x, real_y)

    cv2.setMouseCallback(args.window, mouse_handler)

    while True:
        cv2.imshow(args.window, collector.render_with_instructions())
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            collector.reset()
        if key == ord("u"):
            collector.undo()
        if key in (ord("+"), ord("=")):
            collector.zoom_in()
        if key == ord("-"):
            collector.zoom_out()
        if key == 81:  # left arrow
            collector.pan(-1, 0)
        if key == 83:  # right arrow
            collector.pan(1, 0)
        if key == 82:  # up arrow
            collector.pan(0, -1)
        if key == 84:  # down arrow
            collector.pan(0, 1)
        if key == ord("s"):
            if len(collector.points) == 4:
                save_points(collector.points, json_path, (image.shape[1], image.shape[0]))
                save_annotated_image(collector.annotated_image(), annotated_path)
            else:
                print("Need 4 points before saving.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
