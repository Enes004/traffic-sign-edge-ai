#!/usr/bin/env python3
"""Render a contact sheet of annotated YOLO images for visual validation.

The script reads the prepared dataset in ``data/gtsdb`` (images, labels, and
``dataset.yaml``) without modifying it, draws every bounding box with its class
name, and writes a deterministic contact sheet to
``results/dataset_validation.jpg``.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "data" / "gtsdb"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "dataset_validation.jpg"
DEFAULT_SEED = 42
DEFAULT_COUNT = 12
SPLIT_NAMES = ("train", "val", "test")
COLUMNS = 4
THUMB_SIZE = (400, 235)
CELL_WIDTH = 400
CELL_HEIGHT = 268
GAP = 18
PADDING = 20
HEADER_HEIGHT = 52
BACKGROUND = (24, 24, 27)
TEXT_COLOR = (235, 235, 235)
MUTED_TEXT_COLOR = (170, 170, 175)
CLASS_COLORS = (
    (198, 40, 40),
    (239, 108, 0),
    (21, 101, 192),
    (106, 27, 154),
)

try:
    RESAMPLE_FILTER = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1 or Pillow missing
    RESAMPLE_FILTER = getattr(Image, "LANCZOS", None)


@dataclass(frozen=True)
class Box:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class Sample:
    split: str
    stem: str
    image_path: Path
    boxes: Tuple[Box, ...]


class VisualizationError(RuntimeError):
    """Raised when the dataset cannot be visualized safely."""


def _require_pillow() -> None:
    if Image is None or ImageDraw is None or ImageFont is None:
        raise VisualizationError(
            "Pillow is required to draw labels and write the contact sheet. "
            "Install it with: python3 -m pip install Pillow"
        )


def load_class_names(dataset_root: Path) -> Dict[int, str]:
    """Read the numeric ``names`` mapping from the prepared dataset.yaml."""
    path = Path(dataset_root) / "dataset.yaml"
    if not path.is_file():
        raise VisualizationError(f"Missing dataset config: {path}")

    names: Dict[int, str] = {}
    in_names = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not in_names:
            if not stripped.startswith("names:"):
                continue
            inline = stripped[len("names:") :].strip()
            if not inline:
                in_names = True
                continue
            if inline.startswith("[") and inline.endswith("]"):
                items = [
                    item.strip().strip("'\"")
                    for item in inline[1:-1].split(",")
                    if item.strip()
                ]
                names = dict(enumerate(items))
            continue
        if not line[0].isspace():
            break
        key, separator, value = stripped.partition(":")
        if not separator:
            continue
        try:
            class_id = int(key.strip())
        except ValueError:
            continue
        names[class_id] = value.strip().strip("'\"")

    if not names:
        raise VisualizationError(f"No class names found in {path}")
    expected_ids = list(range(len(names)))
    if sorted(names) != expected_ids:
        raise VisualizationError(
            f"Class IDs in {path} must be contiguous from 0, found {sorted(names)}"
        )
    return names


def parse_label_file(path: Path, class_count: int) -> List[Box]:
    """Parse a YOLO label file, returning an empty list for negative samples."""
    boxes: List[Box] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise VisualizationError(f"Cannot read label file {path}: {exc}") from exc

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise VisualizationError(
                f"{path}:{line_number}: expected 5 fields, found {len(parts)}"
            )
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = (float(value) for value in parts[1:])
        except ValueError as exc:
            raise VisualizationError(
                f"{path}:{line_number}: non-numeric YOLO fields"
            ) from exc
        if not 0 <= class_id < class_count:
            raise VisualizationError(
                f"{path}:{line_number}: class ID {class_id} is outside "
                f"0-{class_count - 1}"
            )
        if not (
            0.0 <= x_center <= 1.0
            and 0.0 <= y_center <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
        ):
            raise VisualizationError(
                f"{path}:{line_number}: normalized box is outside [0, 1]"
            )
        boxes.append(Box(class_id, x_center, y_center, width, height))
    return boxes


def collect_annotated_samples(
    dataset_root: Path, class_count: int
) -> List[Sample]:
    """Collect image/label pairs whose label files contain at least one box."""
    samples: List[Sample] = []
    for split in SPLIT_NAMES:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise VisualizationError(
                f"Missing prepared split directories for {split!r} under "
                f"{dataset_root}"
            )
        for label_path in sorted(label_dir.glob("*.txt")):
            boxes = parse_label_file(label_path, class_count)
            if not boxes:
                continue
            image_path = image_dir / f"{label_path.stem}.jpg"
            if not image_path.is_file():
                raise VisualizationError(
                    f"Label {label_path} has no matching image {image_path}"
                )
            samples.append(
                Sample(
                    split=split,
                    stem=label_path.stem,
                    image_path=image_path,
                    boxes=tuple(boxes),
                )
            )
    return samples


def select_samples(samples: Sequence[Sample], count: int, seed: int) -> List[Sample]:
    """Pick samples deterministically, balancing across train/val/test."""
    if count <= 0:
        raise VisualizationError("--count must be greater than zero")
    if not samples:
        raise VisualizationError("No annotated images found in the prepared dataset")

    pools: Dict[str, List[Sample]] = {split: [] for split in SPLIT_NAMES}
    for sample in sorted(samples, key=lambda item: (item.split, item.stem)):
        pools[sample.split].append(sample)

    rng = random.Random(seed)
    for pool in pools.values():
        rng.shuffle(pool)

    selected: List[Sample] = []
    positions = {split: 0 for split in SPLIT_NAMES}
    while len(selected) < count:
        progressed = False
        for split in SPLIT_NAMES:
            pool = pools[split]
            if positions[split] >= len(pool):
                continue
            selected.append(pool[positions[split]])
            positions[split] += 1
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            break

    if not selected:
        raise VisualizationError("Could not select any annotated images")
    if len(selected) < count:
        print(
            f"warning: only {len(selected)} annotated images are available "
            f"(requested {count})",
            file=sys.stderr,
        )
    rng.shuffle(selected)
    return selected


def _load_font(size: int, bold: bool = False) -> "ImageFont.ImageFont":
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = (
        name,
        f"/usr/share/fonts/truetype/dejavu/{name}",
        f"/usr/share/fonts/dejavu/{name}",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw: "ImageDraw.ImageDraw", text: str, font) -> Tuple[int, int]:
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    except AttributeError:  # Pillow < 8
        return draw.textsize(text, font=font)


def _draw_boxes(
    canvas: "Image.Image",
    thumb_size: Tuple[int, int],
    boxes: Sequence[Box],
    class_names: Dict[int, str],
    offset: Tuple[int, int],
    font,
) -> None:
    draw = ImageDraw.Draw(canvas)
    thumb_width, thumb_height = thumb_size
    offset_x, offset_y = offset

    for box in boxes:
        color = CLASS_COLORS[box.class_id % len(CLASS_COLORS)]
        left = offset_x + int((box.x_center - box.width / 2.0) * thumb_width)
        top = offset_y + int((box.y_center - box.height / 2.0) * thumb_height)
        right = offset_x + int((box.x_center + box.width / 2.0) * thumb_width)
        bottom = offset_y + int((box.y_center + box.height / 2.0) * thumb_height)
        left = max(offset_x, min(left, offset_x + thumb_width - 1))
        top = max(offset_y, min(top, offset_y + thumb_height - 1))
        right = max(left + 1, min(right, offset_x + thumb_width - 1))
        bottom = max(top + 1, min(bottom, offset_y + thumb_height - 1))

        draw.rectangle((left, top, right, bottom), outline=color, width=3)

        label = class_names[box.class_id]
        text_width, text_height = _text_size(draw, label, font)
        label_x = left
        label_y = top - text_height - 6
        if label_y < offset_y:
            label_y = top + 2
        if label_x + text_width + 8 > offset_x + thumb_width:
            label_x = max(offset_x, offset_x + thumb_width - text_width - 8)
        draw.rectangle(
            (
                label_x,
                label_y,
                label_x + text_width + 8,
                label_y + text_height + 6,
            ),
            fill=color,
        )
        draw.text((label_x + 4, label_y + 2), label, font=font, fill=(255, 255, 255))


def render_contact_sheet(
    samples: Sequence[Sample],
    class_names: Dict[int, str],
    output_path: Path,
    seed: int,
    requested_count: int,
) -> None:
    columns = min(COLUMNS, len(samples))
    rows = math.ceil(len(samples) / columns)
    sheet_width = 2 * PADDING + columns * CELL_WIDTH + (columns - 1) * GAP
    sheet_height = (
        2 * PADDING
        + HEADER_HEIGHT
        + rows * CELL_HEIGHT
        + (rows - 1) * GAP
    )

    sheet = Image.new("RGB", (sheet_width, sheet_height), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    title_font = _load_font(24, bold=True)
    subtitle_font = _load_font(15)
    caption_font = _load_font(15)
    label_font = _load_font(14, bold=True)

    draw.text(
        (PADDING, PADDING),
        "GTSDB YOLO label validation",
        font=title_font,
        fill=TEXT_COLOR,
    )
    split_counts = {
        split: sum(1 for sample in samples if sample.split == split)
        for split in SPLIT_NAMES
    }
    subtitle = "  ".join(
        f"{split}={split_counts[split]}" for split in SPLIT_NAMES
    )
    subtitle += f"  seed={seed}"
    subtitle_width, subtitle_height = _text_size(draw, subtitle, subtitle_font)
    draw.text(
        (sheet_width - PADDING - subtitle_width, PADDING + 8),
        subtitle,
        font=subtitle_font,
        fill=MUTED_TEXT_COLOR,
    )

    grid_top = PADDING + HEADER_HEIGHT
    for index, sample in enumerate(samples):
        row, column = divmod(index, columns)
        cell_x = PADDING + column * (CELL_WIDTH + GAP)
        cell_y = grid_top + row * (CELL_HEIGHT + GAP)

        with Image.open(sample.image_path) as source:
            thumbnail = source.convert("RGB")
        thumbnail.thumbnail(THUMB_SIZE, RESAMPLE_FILTER)

        offset_x = cell_x + (CELL_WIDTH - thumbnail.width) // 2
        offset_y = cell_y + (THUMB_SIZE[1] - thumbnail.height) // 2
        sheet.paste(thumbnail, (offset_x, offset_y))
        _draw_boxes(
            sheet,
            thumbnail.size,
            sample.boxes,
            class_names,
            (offset_x, offset_y),
            label_font,
        )

        box_word = "box" if len(sample.boxes) == 1 else "boxes"
        caption = (
            f"{sample.split}/{sample.stem}  |  "
            f"{len(sample.boxes)} {box_word}"
        )
        draw.text(
            (cell_x, cell_y + THUMB_SIZE[1] + 10),
            caption,
            font=caption_font,
            fill=TEXT_COLOR,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=90, optimize=False)
    if len(samples) < requested_count:
        print(
            f"Rendered {len(samples)} of {requested_count} requested samples",
            file=sys.stderr,
        )


def visualize(
    dataset_root: Path,
    output_path: Path,
    seed: int = DEFAULT_SEED,
    count: int = DEFAULT_COUNT,
) -> Path:
    _require_pillow()
    dataset_root = Path(dataset_root).resolve()
    output_path = Path(output_path).resolve()
    class_names = load_class_names(dataset_root)
    samples = collect_annotated_samples(dataset_root, len(class_names))
    selected = select_samples(samples, count, seed)
    render_contact_sheet(selected, class_names, output_path, seed, count)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Draw YOLO labels on a random contact sheet of annotated images "
            "from the prepared GTSDB dataset. The dataset is never modified."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"prepared dataset root (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output JPEG path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"random seed for sample selection (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"number of annotated images to show (default: {DEFAULT_COUNT})",
    )
    return parser


def main(argv: Sequence[str] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output_path = visualize(
            dataset_root=args.dataset,
            output_path=args.output,
            seed=args.seed,
            count=args.count,
        )
    except (OSError, VisualizationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
