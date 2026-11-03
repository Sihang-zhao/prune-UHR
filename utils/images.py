"""Deterministic image preparation for high-resolution VQA inputs."""
from __future__ import annotations

import io
import math
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


_MAX_SOURCE_PIXELS = 50_000_000


def _normalise_open_image(image: Image.Image, max_source_pixels: int = _MAX_SOURCE_PIXELS) -> Image.Image:
    pixels = image.width * image.height
    max_source_pixels = max(1, int(max_source_pixels))
    if pixels > max_source_pixels:
        scale = math.sqrt(max_source_pixels / pixels)
        target = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        # JPEG decoders can avoid materialising the full source when draft size
        # is requested before EXIF handling and the final resize.
        try:
            image.draft("RGB", target)
        except (AttributeError, ValueError):
            pass
        image = image.resize(target, Image.Resampling.LANCZOS)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def _as_rgb(value: Any, max_source_pixels: int = _MAX_SOURCE_PIXELS) -> Image.Image:
    if isinstance(value, Image.Image):
        return _normalise_open_image(value, max_source_pixels)
    if isinstance(value, (bytes, bytearray)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(value)) as image:
                return _normalise_open_image(image, max_source_pixels)
    if isinstance(value, Mapping):
        nested = value.get("bytes", value.get("path", value.get("file", value.get("filename"))))
        if nested is not None:
            return _as_rgb(nested, max_source_pixels)
    if isinstance(value, (str, Path)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(Path(value).expanduser()) as image:
                return _normalise_open_image(image, max_source_pixels)
    raise TypeError(f"unsupported image value: {type(value).__name__}")


def _resize_pad(image: Image.Image, size: int) -> Image.Image:
    size = max(1, int(size))
    return ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)


def _positions(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    stride = max(1, stride)
    values = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if values[-1] != last:
        values.append(last)
    return values


def _evenly_spaced(count: int, budget: int) -> list[int]:
    if budget >= count:
        return list(range(count))
    if budget <= 1:
        return [count // 2]
    return sorted({round(index * (count - 1) / (budget - 1)) for index in range(budget)})


def _contact_sheet(images: Sequence[Image.Image], size: int = 448) -> Image.Image:
    if not images:
        raise ValueError("cannot build a contact sheet from no images")
    thumbnails = [_resize_pad(image, max(32, size // 4)) for image in images[:64]]
    cell = max(image.width for image in thumbnails), max(image.height for image in thumbnails)
    columns = max(1, math.ceil(math.sqrt(len(thumbnails))))
    rows = math.ceil(len(thumbnails) / columns)
    sheet = Image.new("RGB", (columns * cell[0], rows * cell[1]), "white")
    for index, image in enumerate(thumbnails):
        x = (index % columns) * cell[0]
        y = (index // columns) * cell[1]
        sheet.paste(ImageOps.contain(image, cell, Image.Resampling.LANCZOS), (x, y))
    return _resize_pad(sheet, size)


def tile_image(
    image: Image.Image,
    *,
    patch_size: int,
    stride: int,
    max_patches: int,
    include_global: bool,
    output_size: int,
) -> list[Image.Image]:
    image = image.convert("RGB")
    patch_size = max(32, int(patch_size))
    stride = max(1, int(stride))
    max_patches = max(1, int(max_patches))
    views: list[Image.Image] = []
    if include_global and (image.width > patch_size or image.height > patch_size):
        views.append(_resize_pad(image, output_size))
    xs, ys = _positions(image.width, patch_size, stride), _positions(image.height, patch_size, stride)
    budget = max_patches - len(views)
    if budget <= 0:
        return views[:max_patches]
    row_count = min(len(ys), max(1, int(math.sqrt(budget))))
    col_count = min(len(xs), max(1, budget // row_count))
    while row_count * col_count < budget and (row_count < len(ys) or col_count < len(xs)):
        if col_count < len(xs):
            col_count += 1
        elif row_count < len(ys):
            row_count += 1
    row_indices = _evenly_spaced(len(ys), row_count)
    col_indices = _evenly_spaced(len(xs), col_count)
    selected = [(row, col) for row in row_indices for col in col_indices][:budget]
    for row, col in selected:
        x, y = xs[col], ys[row]
        views.append(image.crop((x, y, x + patch_size, y + patch_size)).convert("RGB"))
    if not views:
        views.append(_resize_pad(image, output_size))
    return views[:max_patches]


def prepare_views(
    value: Any,
    *,
    root: str | None = None,
    max_patches: int = 64,
    include_global: bool = True,
    selection: str = "uniform",
    tile_single: bool = True,
    patch_size: int = 448,
    patch_stride: int = 336,
    global_size: int = 448,
    max_source_pixels: int = _MAX_SOURCE_PIXELS,
) -> list[Image.Image]:
    """Build a bounded, ordered list of global and local views for Qwen.

    Embedded XLRS images are already patch sequences, so their original order is
    retained and uniformly sampled when the budget is exceeded. A contact sheet
    gives Qwen a scene-level view before the local patches. Standalone images can
    optionally be tiled with the same budget.
    """
    if isinstance(value, Mapping) and "__hf__" in value:
        raise TypeError("HF image references must be resolved before prepare_views")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        patch_values = list(value)
        if root:
            root_path = Path(root).expanduser()
            patch_values = [
                root_path / item
                if isinstance(item, (str, Path)) and not Path(item).expanduser().is_absolute()
                else item
                for item in patch_values
            ]
        if not patch_values:
            raise ValueError("image patch sequence is empty")
        max_patches = max(1, int(max_patches))
        if len(patch_values) == 1:
            # XLRS stores many standalone high-resolution images as a
            # one-element List(Image).  Treat that representation exactly like
            # a standalone image so --max-image-patches can add local views
            # when requested; with max_patches=1 this still returns one global
            # view and preserves the low-memory/reference protocol.
            single = _as_rgb(patch_values[0], max_source_pixels)
            if max_patches == 1 and include_global:
                # Preserve the full-resolution global view.  Routing this
                # through tile_image would unnecessarily shrink it to the
                # 448-pixel display size before Qwen's own pixel budget is
                # applied.
                return [single]
            if tile_single and (single.width > patch_size or single.height > patch_size):
                return tile_image(
                    single,
                    patch_size=patch_size,
                    stride=patch_stride,
                    max_patches=max_patches,
                    include_global=include_global,
                    output_size=global_size,
                )
            return [single]
        local_budget = max(1, max_patches - (1 if include_global else 0))
        if selection == "head":
            indices = list(range(min(local_budget, len(patch_values))))
        elif selection == "tail":
            indices = list(range(max(0, len(patch_values) - local_budget), len(patch_values)))
        else:
            indices = _evenly_spaced(len(patch_values), local_budget)
        decoded: dict[int, Image.Image] = {}

        def get_patch(index: int) -> Image.Image:
            if index not in decoded:
                decoded[index] = _as_rgb(patch_values[index], max_source_pixels)
            return decoded[index]

        if include_global:
            sheet_indices = _evenly_spaced(len(patch_values), min(64, len(patch_values)))
            sheet_source = [get_patch(index) for index in sheet_indices]
            selected = [get_patch(index) for index in indices]
            return [_contact_sheet(sheet_source, global_size), *selected][:max_patches]
        return [get_patch(index) for index in indices[:max_patches]]

    image = _as_rgb(
        value if not isinstance(value, (str, Path))
        else (Path(root).expanduser() / value if root and not Path(value).expanduser().is_absolute() else value),
        max_source_pixels,
    )
    if tile_single and (image.width > patch_size or image.height > patch_size):
        return tile_image(
            image,
            patch_size=patch_size,
            stride=patch_stride,
            max_patches=max_patches,
            include_global=include_global,
            output_size=global_size,
        )
    return [image]


__all__ = ["prepare_views", "tile_image"]
