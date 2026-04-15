from __future__ import annotations

import dataclasses
import json
import math
import time
import uuid
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from crimson_texture_forge.core.common import run_process_with_cancellation
from crimson_texture_forge.core.pipeline import build_preview_png_command, parse_dds
from crimson_texture_forge.core.upscale_profiles import infer_texture_semantics, is_technical_texture_type
from crimson_texture_forge.models import (
    DdsInfo,
    TextureEditorAdjustmentLayer,
    TextureEditorDocument,
    TextureEditorFloatingSelection,
    TextureEditorHistoryEntry,
    TextureEditorLayer,
    TextureEditorSelection,
    TextureEditorSourceBinding,
    TextureEditorToolSettings,
)

_PROJECT_VERSION = 1
_VISIBLE_TEXTURE_TYPES = {"color", "ui", "emissive", "impostor", "unknown"}


def make_texture_editor_workspace_root(base_dir: Path) -> Path:
    root = base_dir / "texture_editor_workspace"
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_texture_editor_document_root(workspace_root: Path, title: str) -> Path:
    safe_title = _safe_slug(title or "texture_editor")
    root = workspace_root / f"{safe_title}_{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_slug(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "texture_editor"


def _normalize_hex(value: str, fallback: str) -> str:
    text = value.strip().upper()
    if not text:
        return fallback.upper()
    if not text.startswith("#"):
        text = f"#{text}"
    if len(text) != 7:
        return fallback.upper()
    try:
        int(text[1:], 16)
    except Exception:
        return fallback.upper()
    return text


def _parse_hex_rgb(value: str, fallback: str = "#C85A30") -> Tuple[int, int, int]:
    text = _normalize_hex(value, fallback)
    return (int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16))


def _new_layer_id() -> str:
    return uuid.uuid4().hex[:12]


def _load_rgba_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        return np.asarray(rgba, dtype=np.uint8).copy()


def save_rgba_array_png(array: np.ndarray, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8), "RGBA").save(output_path, format="PNG")
    return output_path


def normalize_texture_editor_source_to_png(
    source_path: Path,
    *,
    texconv_path: Optional[Path],
    output_dir: Path,
    output_stem: str = "",
) -> Path:
    resolved = source_path.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = resolved.suffix.lower()
    stem = output_stem.strip() or resolved.stem
    output_path = output_dir / f"{stem}.png"
    if suffix == ".dds":
        if texconv_path is None or not texconv_path.exists():
            raise ValueError("texconv.exe is required to open DDS files in Texture Editor.")
        cmd = build_preview_png_command(texconv_path, resolved, output_dir)
        return_code, stdout, stderr = run_process_with_cancellation(cmd, stop_event=None)
        if return_code != 0:
            detail = stderr.strip() or stdout.strip() or f"texconv failed with exit code {return_code}"
            raise ValueError(f"Could not normalize DDS for Texture Editor: {detail}")
        if output_path.exists():
            return output_path
        png_candidates = sorted(output_dir.glob("*.png"))
        if not png_candidates:
            raise ValueError(f"texconv did not create a PNG for {resolved.name}")
        return png_candidates[0]
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tga"}:
        raise ValueError(f"Unsupported texture source for Texture Editor: {resolved.suffix}")
    with Image.open(resolved) as image:
        image.convert("RGBA").save(output_path, format="PNG")
    return output_path


def derive_texture_editor_binding(
    source_path: Path,
    *,
    binding: Optional[TextureEditorSourceBinding] = None,
) -> Tuple[TextureEditorSourceBinding, Optional[DdsInfo]]:
    resolved = source_path.expanduser().resolve()
    source_binding = dataclasses.replace(binding) if binding is not None else TextureEditorSourceBinding()
    source_binding.source_path = str(resolved)
    if not source_binding.source_identity_path:
        source_binding.source_identity_path = str(resolved)
    if not source_binding.display_name:
        source_binding.display_name = resolved.name
    dds_info: Optional[DdsInfo] = None
    original_dds = Path(source_binding.original_dds_path).expanduser() if source_binding.original_dds_path else None
    try:
        if original_dds and original_dds.exists():
            dds_info = parse_dds(original_dds)
            source_binding.original_texconv_format = dds_info.texconv_format
    except Exception:
        dds_info = None

    semantic_path = source_binding.relative_path or source_binding.archive_relative_path or resolved.name
    semantic = infer_texture_semantics(
        semantic_path,
        original_texconv_format=source_binding.original_texconv_format,
    )
    source_binding.texture_type = semantic.texture_type
    source_binding.semantic_subtype = semantic.semantic_subtype
    if is_technical_texture_type(semantic.texture_type):
        source_binding.technical_warning = (
            f"This looks like a technical texture ({semantic.texture_type}/{semantic.semantic_subtype}). "
            "Painting or recoloring it may break the intended data."
        )
    elif semantic.texture_type not in _VISIBLE_TEXTURE_TYPES:
        source_binding.technical_warning = (
            f"This texture is classified as {semantic.texture_type}/{semantic.semantic_subtype}. "
            "Texture Editor is primarily intended for visible-color texture work."
        )
    else:
        source_binding.technical_warning = ""
    return source_binding, dds_info


def create_texture_editor_document_from_source(
    source_path: Path,
    *,
    texconv_path: Optional[Path],
    workspace_root: Path,
    binding: Optional[TextureEditorSourceBinding] = None,
) -> Tuple[TextureEditorDocument, Dict[str, np.ndarray], Path]:
    resolved = source_path.expanduser().resolve()
    binding, _dds_info = derive_texture_editor_binding(resolved, binding=binding)
    document_root = build_texture_editor_document_root(workspace_root, resolved.stem)
    normalized_png = normalize_texture_editor_source_to_png(
        resolved,
        texconv_path=texconv_path,
        output_dir=document_root / "normalized",
        output_stem=resolved.stem,
    )
    pixels = _load_rgba_array(normalized_png)
    height, width = pixels.shape[:2]
    layer = TextureEditorLayer(
        layer_id=_new_layer_id(),
        name="Base Layer",
        relative_png_path="layers/base.png",
        visible=True,
        opacity=100,
        thumbnail_cache_key=uuid.uuid4().hex,
    )
    document = TextureEditorDocument(
        title=resolved.stem,
        width=int(width),
        height=int(height),
        workspace_root=document_root,
        active_layer_id=layer.layer_id,
        layers=(layer,),
        source_binding=binding,
        technical_warning=binding.technical_warning,
    )
    layer_pixels = {layer.layer_id: pixels}
    return document, layer_pixels, normalized_png


def _selection_to_dict(selection: TextureEditorSelection) -> Dict[str, object]:
    return {
        "mode": selection.mode,
        "rect": list(selection.rect) if selection.rect else None,
        "polygon_points": [list(point) for point in selection.polygon_points],
        "mask_polygons": [[list(point) for point in polygon] for polygon in selection.mask_polygons],
        "inverted": bool(selection.inverted),
        "feather_radius": int(selection.feather_radius),
    }


def _selection_from_dict(data: Dict[str, object]) -> TextureEditorSelection:
    rect_value = data.get("rect")
    rect: Optional[Tuple[int, int, int, int]] = None
    if isinstance(rect_value, list) and len(rect_value) == 4:
        rect = tuple(int(value) for value in rect_value)  # type: ignore[assignment]
    polygon_points_raw = data.get("polygon_points")
    polygon_points: Tuple[Tuple[float, float], ...] = ()
    if isinstance(polygon_points_raw, list):
        polygon_points = tuple(
            (float(point[0]), float(point[1]))
            for point in polygon_points_raw
            if isinstance(point, list) and len(point) == 2
        )
    mask_polygons_raw = data.get("mask_polygons")
    mask_polygons: Tuple[Tuple[Tuple[float, float], ...], ...] = ()
    if isinstance(mask_polygons_raw, list):
        polygons: List[Tuple[Tuple[float, float], ...]] = []
        for polygon in mask_polygons_raw:
            if not isinstance(polygon, list):
                continue
            points = tuple(
                (float(point[0]), float(point[1]))
                for point in polygon
                if isinstance(point, list) and len(point) == 2
            )
            if len(points) >= 3:
                polygons.append(points)
        mask_polygons = tuple(polygons)
    return TextureEditorSelection(
        mode=str(data.get("mode", "none") or "none"),
        rect=rect,
        polygon_points=polygon_points,
        mask_polygons=mask_polygons,
        inverted=bool(data.get("inverted", False)),
        feather_radius=max(0, int(data.get("feather_radius", 0) or 0)),
    )


def _floating_selection_to_dict(floating: Optional[TextureEditorFloatingSelection]) -> Optional[Dict[str, object]]:
    if floating is None:
        return None
    return {
        "source_layer_id": floating.source_layer_id,
        "label": floating.label,
        "bounds": list(floating.bounds),
        "offset_x": int(floating.offset_x),
        "offset_y": int(floating.offset_y),
        "scale_x": float(floating.scale_x),
        "scale_y": float(floating.scale_y),
        "rotation_degrees": float(floating.rotation_degrees),
        "flip_x": bool(floating.flip_x),
        "flip_y": bool(floating.flip_y),
        "paste_mode": str(floating.paste_mode or "in_place"),
        "committed": bool(floating.committed),
    }


def _floating_selection_from_dict(data: Optional[Dict[str, object]]) -> Optional[TextureEditorFloatingSelection]:
    if not isinstance(data, dict):
        return None
    bounds_raw = data.get("bounds")
    bounds = (0, 0, 0, 0)
    if isinstance(bounds_raw, list) and len(bounds_raw) == 4:
        bounds = tuple(int(value) for value in bounds_raw)  # type: ignore[assignment]
    return TextureEditorFloatingSelection(
        source_layer_id=str(data.get("source_layer_id", "")),
        label=str(data.get("label", "")),
        bounds=bounds,
        offset_x=int(data.get("offset_x", 0) or 0),
        offset_y=int(data.get("offset_y", 0) or 0),
        scale_x=float(data.get("scale_x", 1.0) or 1.0),
        scale_y=float(data.get("scale_y", 1.0) or 1.0),
        rotation_degrees=float(data.get("rotation_degrees", 0.0) or 0.0),
        flip_x=bool(data.get("flip_x", False)),
        flip_y=bool(data.get("flip_y", False)),
        paste_mode=str(data.get("paste_mode", "in_place") or "in_place"),
        committed=bool(data.get("committed", True)),
    )


def _adjustment_layer_to_dict(layer: TextureEditorAdjustmentLayer) -> Dict[str, object]:
    return {
        "layer_id": str(layer.layer_id),
        "name": str(layer.name),
        "adjustment_type": str(layer.adjustment_type),
        "enabled": bool(layer.enabled),
        "opacity": int(layer.opacity),
        "parameters": {str(key): float(value) for key, value in layer.parameters.items()},
        "mask_layer_id": str(layer.mask_layer_id),
        "revision": int(layer.revision),
    }


def _adjustment_layer_from_dict(data: Dict[str, object]) -> TextureEditorAdjustmentLayer:
    parameters_raw = data.get("parameters")
    parameters: Dict[str, float] = {}
    if isinstance(parameters_raw, dict):
        for key, value in parameters_raw.items():
            try:
                parameters[str(key)] = float(value)
            except Exception:
                continue
    return TextureEditorAdjustmentLayer(
        layer_id=str(data.get("layer_id", "")),
        name=str(data.get("name", "Adjustment")),
        adjustment_type=str(data.get("adjustment_type", "levels") or "levels"),
        enabled=bool(data.get("enabled", True)),
        opacity=int(data.get("opacity", 100) or 100),
        parameters=parameters,
        mask_layer_id=str(data.get("mask_layer_id", "")),
        revision=int(data.get("revision", 0) or 0),
    )


def save_texture_editor_project(
    document: TextureEditorDocument,
    layer_pixels: Dict[str, np.ndarray],
    project_path: Path,
    *,
    floating_pixels: Optional[np.ndarray] = None,
) -> TextureEditorDocument:
    project_path = project_path.expanduser().resolve()
    project_path.parent.mkdir(parents=True, exist_ok=True)
    assets_dir = project_path.with_suffix("")
    assets_dir = assets_dir.parent / f"{assets_dir.name}_assets"
    layers_dir = assets_dir / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)

    saved_layers: List[TextureEditorLayer] = []
    saved_masks: Dict[str, str] = {}
    for index, layer in enumerate(document.layers, start=1):
        pixels = layer_pixels.get(layer.layer_id)
        if pixels is None:
            continue
        file_name = f"{index:02d}_{_safe_slug(layer.name)}.png"
        relative_png = PurePosixPath("layers") / file_name
        save_rgba_array_png(pixels, layers_dir / file_name)
        saved_layers.append(
            dataclasses.replace(
                layer,
                relative_png_path=relative_png.as_posix(),
            )
        )
        if layer.mask_layer_id and layer.mask_layer_id in layer_pixels:
            masks_dir = assets_dir / "masks"
            masks_dir.mkdir(parents=True, exist_ok=True)
            mask_name = f"{index:02d}_{_safe_slug(layer.name)}_mask.png"
            save_rgba_array_png(layer_pixels[layer.mask_layer_id], masks_dir / mask_name)
            saved_masks[layer.mask_layer_id] = (PurePosixPath("masks") / mask_name).as_posix()

    payload = {
        "version": _PROJECT_VERSION,
        "title": document.title,
        "width": document.width,
        "height": document.height,
        "active_layer_id": document.active_layer_id,
        "technical_warning": document.technical_warning,
        "last_flattened_png_path": document.last_flattened_png_path,
        "source_binding": dataclasses.asdict(document.source_binding),
        "selection": _selection_to_dict(document.selection),
        "floating_selection": _floating_selection_to_dict(document.floating_selection),
        "adjustment_layers": [_adjustment_layer_to_dict(layer) for layer in document.adjustment_layers],
        "composite_revision": int(document.composite_revision),
        "quick_mask_enabled": bool(document.quick_mask_enabled),
        "edit_red_channel": bool(document.edit_red_channel),
        "edit_green_channel": bool(document.edit_green_channel),
        "edit_blue_channel": bool(document.edit_blue_channel),
        "edit_alpha_channel": bool(document.edit_alpha_channel),
        "masks": saved_masks,
        "floating_pixels_path": "",
        "layers": [dataclasses.asdict(layer) for layer in saved_layers],
    }
    if document.floating_selection is not None and floating_pixels is not None:
        floating_dir = assets_dir / "floating"
        floating_dir.mkdir(parents=True, exist_ok=True)
        floating_name = "floating_selection.png"
        save_rgba_array_png(floating_pixels, floating_dir / floating_name)
        payload["floating_pixels_path"] = (PurePosixPath("floating") / floating_name).as_posix()
    project_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dataclasses.replace(
        document,
        project_path=project_path,
        layers=tuple(saved_layers),
    )


def load_texture_editor_project(
    project_path: Path,
) -> Tuple[TextureEditorDocument, Dict[str, np.ndarray], Optional[np.ndarray]]:
    resolved = project_path.expanduser().resolve()
    data = json.loads(resolved.read_text(encoding="utf-8"))
    assets_dir = resolved.with_suffix("")
    assets_dir = assets_dir.parent / f"{assets_dir.name}_assets"
    layers: List[TextureEditorLayer] = []
    layer_pixels: Dict[str, np.ndarray] = {}
    missing_assets: List[str] = []
    for layer_data in data.get("layers", []):
        layer = TextureEditorLayer(
            layer_id=str(layer_data.get("layer_id", "")),
            name=str(layer_data.get("name", "Layer")),
            relative_png_path=str(layer_data.get("relative_png_path", "")),
            visible=bool(layer_data.get("visible", True)),
            opacity=int(layer_data.get("opacity", 100)),
            blend_mode=str(layer_data.get("blend_mode", "normal") or "normal"),
            offset_x=int(layer_data.get("offset_x", 0) or 0),
            offset_y=int(layer_data.get("offset_y", 0) or 0),
            locked=bool(layer_data.get("locked", False)),
            alpha_locked=bool(layer_data.get("alpha_locked", False)),
            mask_layer_id=str(layer_data.get("mask_layer_id", "")),
            mask_enabled=bool(layer_data.get("mask_enabled", True)),
            revision=int(layer_data.get("revision", 0) or 0),
            thumbnail_cache_key=str(layer_data.get("thumbnail_cache_key", "")),
        )
        png_path = assets_dir / Path(layer.relative_png_path)
        if png_path.exists():
            layer_pixels[layer.layer_id] = _load_rgba_array(png_path)
            layers.append(layer)
        else:
            missing_assets.append(str(png_path))
    masks_raw = data.get("masks") or {}
    if isinstance(masks_raw, dict):
        for mask_layer_id, relative_path in masks_raw.items():
            mask_path = assets_dir / Path(str(relative_path))
            if mask_path.exists():
                try:
                    layer_pixels[str(mask_layer_id)] = _load_rgba_array(mask_path)
                except Exception:
                    continue
            else:
                missing_assets.append(str(mask_path))
    floating_pixels: Optional[np.ndarray] = None
    floating_pixels_path = str(data.get("floating_pixels_path", "") or "").strip()
    if floating_pixels_path:
        floating_path = assets_dir / Path(floating_pixels_path)
        if floating_path.exists():
            floating_pixels = _load_rgba_array(floating_path)
        else:
            missing_assets.append(str(floating_path))
    elif data.get("floating_selection"):
        missing_assets.append("<floating selection pixels>")
    if missing_assets:
        sample = ", ".join(missing_assets[:3])
        if len(missing_assets) > 3:
            sample += ", ..."
        raise FileNotFoundError(f"Texture Editor project is missing required asset files: {sample}")
    source_binding = TextureEditorSourceBinding(**(data.get("source_binding") or {}))
    document = TextureEditorDocument(
        title=str(data.get("title", resolved.stem)),
        width=int(data.get("width", 0)),
        height=int(data.get("height", 0)),
        project_path=resolved,
        workspace_root=assets_dir,
        active_layer_id=str(data.get("active_layer_id", "")),
        layers=tuple(layers),
        source_binding=source_binding,
        selection=_selection_from_dict(data.get("selection") or {}),
        floating_selection=_floating_selection_from_dict(data.get("floating_selection") or None),
        adjustment_layers=tuple(
            _adjustment_layer_from_dict(item)
            for item in (data.get("adjustment_layers") or [])
            if isinstance(item, dict)
        ),
        technical_warning=str(data.get("technical_warning", "")),
        last_flattened_png_path=str(data.get("last_flattened_png_path", "")),
        composite_revision=int(data.get("composite_revision", 0) or 0),
        quick_mask_enabled=bool(data.get("quick_mask_enabled", False)),
        edit_red_channel=bool(data.get("edit_red_channel", True)),
        edit_green_channel=bool(data.get("edit_green_channel", True)),
        edit_blue_channel=bool(data.get("edit_blue_channel", True)),
        edit_alpha_channel=bool(data.get("edit_alpha_channel", True)),
    )
    return document, layer_pixels, floating_pixels


def _selection_mask_to_polygons(mask: np.ndarray) -> Tuple[Tuple[Tuple[float, float], ...], ...]:
    binary = np.asarray(mask > 0, dtype=np.uint8)
    if not np.any(binary):
        return ()
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[Tuple[Tuple[float, float], ...]] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        epsilon = max(0.35, cv2.arcLength(contour, True) * 0.002)
        simplified = cv2.approxPolyDP(contour, epsilon, closed=True)
        points = tuple((float(point[0][0]), float(point[0][1])) for point in simplified)
        if len(points) >= 3:
            polygons.append(points)
    polygons.sort(key=lambda polygon: len(polygon), reverse=True)
    return tuple(polygons)


def _selection_from_mask(
    mask: Optional[np.ndarray],
    *,
    inverted: bool = False,
    feather_radius: int = 0,
) -> TextureEditorSelection:
    if mask is None or not np.any(mask):
        return TextureEditorSelection(inverted=False, feather_radius=max(0, int(feather_radius)))
    polygons = _selection_mask_to_polygons(mask)
    if not polygons:
        return TextureEditorSelection(inverted=False, feather_radius=max(0, int(feather_radius)))
    first_polygon = polygons[0]
    return TextureEditorSelection(
        mode="mask",
        polygon_points=first_polygon,
        mask_polygons=polygons,
        inverted=bool(inverted),
        feather_radius=max(0, int(feather_radius)),
    )


def _combine_selection_masks(
    existing_mask: Optional[np.ndarray],
    incoming_mask: Optional[np.ndarray],
    *,
    combine_mode: str,
) -> Optional[np.ndarray]:
    incoming = np.asarray(incoming_mask, dtype=np.uint8) if incoming_mask is not None else None
    existing = np.asarray(existing_mask, dtype=np.uint8) if existing_mask is not None else None
    mode_key = (combine_mode or "replace").strip().lower()
    if mode_key == "replace" or existing is None:
        return incoming.copy() if incoming is not None else None
    if incoming is None:
        return existing.copy()
    if mode_key == "add":
        return np.maximum(existing, incoming)
    if mode_key == "subtract":
        result = existing.astype(np.int16) - incoming.astype(np.int16)
        return np.clip(result, 0, 255).astype(np.uint8)
    if mode_key == "intersect":
        return np.minimum(existing, incoming)
    return incoming.copy()


def _effective_brush_size(settings: TextureEditorToolSettings) -> float:
    base = max(0.25, float(settings.size))
    mode = (getattr(settings, "size_step_mode", "normal") or "normal").strip().lower()
    if mode == "fine":
        return max(0.25, base * 0.25)
    return base


def _smooth_stroke_points(
    points: Sequence[Tuple[int, int]],
    smoothing: int,
) -> List[Tuple[float, float]]:
    if len(points) <= 2:
        return [(float(x), float(y)) for x, y in points]
    strength = max(0, min(100, int(smoothing)))
    if strength <= 0:
        return [(float(x), float(y)) for x, y in points]
    window = 1 + max(1, int(round((strength / 100.0) * 5.0)))
    output: List[Tuple[float, float]] = []
    for index, _point in enumerate(points):
        if index in {0, len(points) - 1}:
            output.append((float(points[index][0]), float(points[index][1])))
            continue
        x_total = 0.0
        y_total = 0.0
        weight_total = 0.0
        for sample_index in range(max(0, index - window), min(len(points), index + window + 1)):
            distance = abs(sample_index - index)
            weight = float(window + 1 - distance)
            x_total += float(points[sample_index][0]) * weight
            y_total += float(points[sample_index][1]) * weight
            weight_total += weight
        output.append((x_total / max(1.0, weight_total), y_total / max(1.0, weight_total)))
    return output


def _channel_edit_flags(document: TextureEditorDocument) -> Tuple[bool, bool, bool, bool]:
    return (
        bool(getattr(document, "edit_red_channel", True)),
        bool(getattr(document, "edit_green_channel", True)),
        bool(getattr(document, "edit_blue_channel", True)),
        bool(getattr(document, "edit_alpha_channel", True)),
    )


def _apply_channel_edit_locks(
    document: TextureEditorDocument,
    before_region: np.ndarray,
    after_region: np.ndarray,
) -> np.ndarray:
    red_enabled, green_enabled, blue_enabled, alpha_enabled = _channel_edit_flags(document)
    if red_enabled and green_enabled and blue_enabled and alpha_enabled:
        return after_region
    locked = after_region.copy()
    if not red_enabled:
        locked[..., 0] = before_region[..., 0]
    if not green_enabled:
        locked[..., 1] = before_region[..., 1]
    if not blue_enabled:
        locked[..., 2] = before_region[..., 2]
    if not alpha_enabled:
        locked[..., 3] = before_region[..., 3]
    return locked


def _layer_canvas_intersection(
    layer: TextureEditorLayer,
    pixels: np.ndarray,
    document: TextureEditorDocument,
) -> Optional[Tuple[int, int, int, int, int, int, int, int]]:
    layer_h, layer_w = pixels.shape[:2]
    if layer_w <= 0 or layer_h <= 0:
        return None
    x0 = int(layer.offset_x)
    y0 = int(layer.offset_y)
    x1 = x0 + layer_w
    y1 = y0 + layer_h
    dx0 = max(0, x0)
    dy0 = max(0, y0)
    dx1 = min(document.width, x1)
    dy1 = min(document.height, y1)
    if dx1 <= dx0 or dy1 <= dy0:
        return None
    sx0 = dx0 - x0
    sy0 = dy0 - y0
    sx1 = sx0 + (dx1 - dx0)
    sy1 = sy0 + (dy1 - dy0)
    return dx0, dy0, dx1, dy1, sx0, sy0, sx1, sy1


def _blend_rgb_mode(dst_rgb: np.ndarray, src_rgb: np.ndarray, mode: str) -> np.ndarray:
    mode_key = (mode or "normal").strip().lower()
    if mode_key == "multiply":
        return dst_rgb * src_rgb
    if mode_key == "screen":
        return 1.0 - ((1.0 - dst_rgb) * (1.0 - src_rgb))
    if mode_key == "overlay":
        return np.where(
            dst_rgb <= 0.5,
            2.0 * dst_rgb * src_rgb,
            1.0 - (2.0 * (1.0 - dst_rgb) * (1.0 - src_rgb)),
        )
    return src_rgb


def _blend_layer_region(
    dst_region: np.ndarray,
    src_region: np.ndarray,
    *,
    opacity: int,
    mode: str,
) -> np.ndarray:
    dst = dst_region.astype(np.float32) / 255.0
    src = src_region.astype(np.float32) / 255.0
    src_alpha = src[..., 3:4] * max(0.0, min(1.0, float(opacity) / 100.0))
    dst_alpha = dst[..., 3:4]
    blended_rgb = _blend_rgb_mode(dst[..., :3], src[..., :3], mode)
    out_alpha = src_alpha + dst_alpha * (1.0 - src_alpha)
    safe_alpha = np.where(out_alpha > 1e-6, out_alpha, 1.0)
    out_rgb = (blended_rgb * src_alpha + dst[..., :3] * dst_alpha * (1.0 - src_alpha)) / safe_alpha
    out = dst.copy()
    out[..., :3] = np.where(out_alpha > 1e-6, out_rgb, out[..., :3])
    out[..., 3:4] = out_alpha
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


def _apply_mask_to_src_region(src_region: np.ndarray, mask_region: Optional[np.ndarray]) -> np.ndarray:
    if mask_region is None or mask_region.size == 0:
        return src_region
    mask_alpha = mask_region[..., 3:4].astype(np.float32) / 255.0
    masked = src_region.copy().astype(np.float32)
    masked[..., 3:4] *= mask_alpha
    return np.clip(np.round(masked), 0, 255).astype(np.uint8)


def _build_curves_lut(shadows: float, midtones: float, highlights: float) -> np.ndarray:
    xs = np.arange(256, dtype=np.float32)
    normalized = xs / 255.0
    shadow_bias = max(-1.0, min(1.0, shadows / 100.0))
    mid_bias = max(-1.0, min(1.0, midtones / 100.0))
    highlight_bias = max(-1.0, min(1.0, highlights / 100.0))
    curve = normalized.copy()
    curve += shadow_bias * ((1.0 - normalized) ** 2) * 0.25
    curve += mid_bias * (1.0 - np.abs((normalized * 2.0) - 1.0)) * 0.30
    curve += highlight_bias * (normalized ** 2) * 0.25
    return np.clip(np.round(curve * 255.0), 0, 255).astype(np.uint8)


def _apply_adjustment_to_rgba(
    rgba: np.ndarray,
    adjustment: TextureEditorAdjustmentLayer,
    *,
    mask_region: Optional[np.ndarray] = None,
) -> np.ndarray:
    if rgba.size == 0 or not adjustment.enabled or adjustment.opacity <= 0:
        return rgba
    opacity = max(0.0, min(1.0, adjustment.opacity / 100.0))
    source = rgba.astype(np.uint8)
    result = source.copy()
    params = adjustment.parameters
    adj_type = (adjustment.adjustment_type or "").strip().lower()
    rgb = source[..., :3]
    alpha = source[..., 3:4]
    if adj_type == "hue_saturation":
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        hue_shift = float(params.get("hue", 0.0))
        sat_shift = float(params.get("saturation", 0.0))
        light_shift = float(params.get("lightness", 0.0))
        hsv[..., 0] = np.mod(hsv[..., 0] + (hue_shift / 2.0), 180.0)
        hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + (sat_shift / 100.0)), 0.0, 255.0)
        hsv[..., 2] = np.clip(hsv[..., 2] * (1.0 + (light_shift / 100.0)), 0.0, 255.0)
        adjusted_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    elif adj_type == "curves":
        lut = _build_curves_lut(
            float(params.get("shadows", 0.0)),
            float(params.get("midtones", 0.0)),
            float(params.get("highlights", 0.0)),
        )
        adjusted_rgb = cv2.LUT(rgb, lut)
    else:
        black = max(0.0, min(254.0, float(params.get("black", 0.0))))
        white = max(1.0, min(255.0, float(params.get("white", 255.0))))
        white = max(white, black + 1.0)
        gamma = max(0.1, min(4.0, float(params.get("gamma", 1.0))))
        out_black = max(0.0, min(254.0, float(params.get("output_black", 0.0))))
        out_white = max(out_black + 1.0, min(255.0, float(params.get("output_white", 255.0))))
        normalized = np.clip((rgb.astype(np.float32) - black) / max(1.0, white - black), 0.0, 1.0)
        leveled = np.power(normalized, 1.0 / gamma)
        adjusted_rgb = np.clip(
            np.round(out_black + (leveled * (out_white - out_black))),
            0.0,
            255.0,
        ).astype(np.uint8)
    adjusted = np.concatenate([adjusted_rgb, alpha], axis=2)
    if mask_region is not None and mask_region.size > 0:
        opacity *= 1.0
        mask_alpha = mask_region[..., 3:4].astype(np.float32) / 255.0
    else:
        mask_alpha = None
    blended = source.astype(np.float32)
    adjusted_f = adjusted.astype(np.float32)
    if mask_alpha is None:
        weight = opacity
    else:
        weight = opacity * mask_alpha
    blended[..., :3] = np.clip(
        (source[..., :3].astype(np.float32) * (1.0 - weight)) + (adjusted_f[..., :3] * weight),
        0.0,
        255.0,
    )
    result[..., :3] = np.round(blended[..., :3]).astype(np.uint8)
    return result


def _flatten_texture_editor_raster_layers(
    document: TextureEditorDocument,
    layer_pixels: Dict[str, np.ndarray],
) -> np.ndarray:
    base = np.zeros((document.height, document.width, 4), dtype=np.uint8)
    for layer in document.layers:
        pixels = layer_pixels.get(layer.layer_id)
        if pixels is None or not layer.visible:
            continue
        intersection = _layer_canvas_intersection(layer, pixels, document)
        if intersection is None:
            continue
        dx0, dy0, dx1, dy1, sx0, sy0, sx1, sy1 = intersection
        dst_region = base[dy0:dy1, dx0:dx1]
        src_region = pixels[sy0:sy1, sx0:sx1]
        if layer.mask_layer_id and layer.mask_enabled:
            mask_pixels = layer_pixels.get(layer.mask_layer_id)
            if mask_pixels is not None:
                src_region = _apply_mask_to_src_region(src_region, mask_pixels[sy0:sy1, sx0:sx1])
        base[dy0:dy1, dx0:dx1] = _blend_layer_region(
            dst_region,
            src_region,
            opacity=layer.opacity,
            mode=layer.blend_mode,
        )
    return base


def flatten_texture_editor_layers(
    document: TextureEditorDocument,
    layer_pixels: Dict[str, np.ndarray],
) -> np.ndarray:
    base = _flatten_texture_editor_raster_layers(document, layer_pixels)
    if not document.adjustment_layers:
        return base
    result = base
    for adjustment in document.adjustment_layers:
        mask_region = layer_pixels.get(adjustment.mask_layer_id) if adjustment.mask_layer_id else None
        result = _apply_adjustment_to_rgba(result, adjustment, mask_region=mask_region)
    return result


def export_texture_editor_flattened_png(
    document: TextureEditorDocument,
    layer_pixels: Dict[str, np.ndarray],
    output_path: Path,
) -> Path:
    flattened = flatten_texture_editor_layers(document, layer_pixels)
    return save_rgba_array_png(flattened, output_path.expanduser().resolve())


def build_texture_editor_selection_mask(
    width: int,
    height: int,
    selection: TextureEditorSelection,
) -> Optional[np.ndarray]:
    width = max(0, int(width))
    height = max(0, int(height))
    if width <= 0 or height <= 0 or selection.mode == "none":
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    if selection.mask_polygons:
        for polygon_points in selection.mask_polygons:
            points = np.asarray(polygon_points, dtype=np.float32)
            if len(points) < 3:
                continue
            scale_factor = 4
            min_x = max(0, int(math.floor(float(np.min(points[:, 0])))) - 2)
            min_y = max(0, int(math.floor(float(np.min(points[:, 1])))) - 2)
            max_x = min(width, int(math.ceil(float(np.max(points[:, 0])))) + 3)
            max_y = min(height, int(math.ceil(float(np.max(points[:, 1])))) + 3)
            if max_x <= min_x or max_y <= min_y:
                continue
            patch_width = max_x - min_x
            patch_height = max_y - min_y
            supersampled = np.zeros((patch_height * scale_factor, patch_width * scale_factor), dtype=np.uint8)
            shifted = np.empty_like(points)
            shifted[:, 0] = (points[:, 0] - float(min_x)) * float(scale_factor)
            shifted[:, 1] = (points[:, 1] - float(min_y)) * float(scale_factor)
            polygon = np.round(shifted).astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(supersampled, [polygon], 255, lineType=cv2.LINE_AA)
            antialiased_patch = cv2.resize(
                supersampled,
                (patch_width, patch_height),
                interpolation=cv2.INTER_AREA,
            )
            current_patch = mask[min_y:max_y, min_x:max_x]
            mask[min_y:max_y, min_x:max_x] = np.maximum(current_patch, antialiased_patch)
    elif selection.mode == "rect" and selection.rect is not None:
        x, y, w, h = selection.rect
        x0 = max(0, min(width, int(x)))
        y0 = max(0, min(height, int(y)))
        x1 = max(x0, min(width, int(x + w)))
        y1 = max(y0, min(height, int(y + h)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    elif selection.mode == "lasso" and selection.polygon_points:
        points = np.asarray(selection.polygon_points, dtype=np.float32)
        if len(points) >= 3:
            scale_factor = 4
            min_x = max(0, int(math.floor(float(np.min(points[:, 0])))) - 2)
            min_y = max(0, int(math.floor(float(np.min(points[:, 1])))) - 2)
            max_x = min(width, int(math.ceil(float(np.max(points[:, 0])))) + 3)
            max_y = min(height, int(math.ceil(float(np.max(points[:, 1])))) + 3)
            if max_x > min_x and max_y > min_y:
                patch_width = max_x - min_x
                patch_height = max_y - min_y
                supersampled = np.zeros((patch_height * scale_factor, patch_width * scale_factor), dtype=np.uint8)
                shifted = np.empty_like(points)
                shifted[:, 0] = (points[:, 0] - float(min_x)) * float(scale_factor)
                shifted[:, 1] = (points[:, 1] - float(min_y)) * float(scale_factor)
                polygon = np.round(shifted).astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(supersampled, [polygon], 255, lineType=cv2.LINE_AA)
                antialiased_patch = cv2.resize(
                    supersampled,
                    (patch_width, patch_height),
                    interpolation=cv2.INTER_AREA,
                )
                current_patch = mask[min_y:max_y, min_x:max_x]
                mask[min_y:max_y, min_x:max_x] = np.maximum(current_patch, antialiased_patch)
    if not np.any(mask):
        return None
    feather_radius = max(0, int(selection.feather_radius))
    if feather_radius > 0:
        kernel = max(3, feather_radius * 2 + 1)
        mask = cv2.GaussianBlur(mask, (kernel, kernel), sigmaX=max(0.8, feather_radius / 2.0))
    if selection.inverted:
        mask = 255 - mask
    return mask if np.any(mask) else None


def clear_texture_editor_selection(document: TextureEditorDocument) -> TextureEditorDocument:
    current = document.selection
    return dataclasses.replace(
        document,
        selection=TextureEditorSelection(
            inverted=False,
            feather_radius=max(0, int(current.feather_radius)),
        ),
        floating_selection=None,
    )


def apply_texture_editor_rect_selection(
    document: TextureEditorDocument,
    rect: Tuple[int, int, int, int],
    *,
    combine_mode: str = "replace",
) -> TextureEditorDocument:
    x, y, w, h = rect
    incoming = np.zeros((document.height, document.width), dtype=np.uint8)
    x0 = max(0, min(document.width, int(x)))
    y0 = max(0, min(document.height, int(y)))
    x1 = max(x0, min(document.width, int(x + w)))
    y1 = max(y0, min(document.height, int(y + h)))
    if x1 > x0 and y1 > y0:
        incoming[y0:y1, x0:x1] = 255
    existing = build_texture_editor_selection_mask(document.width, document.height, document.selection)
    combined = _combine_selection_masks(existing, incoming, combine_mode=combine_mode)
    return dataclasses.replace(
        document,
        selection=_selection_from_mask(
            combined,
            feather_radius=max(0, int(document.selection.feather_radius)),
        ),
    )


def apply_texture_editor_lasso_selection(
    document: TextureEditorDocument,
    polygon_points: Sequence[Tuple[float, float]],
    *,
    combine_mode: str = "replace",
) -> TextureEditorDocument:
    incoming = build_texture_editor_selection_mask(
        document.width,
        document.height,
        TextureEditorSelection(
            mode="lasso",
            polygon_points=tuple((float(x), float(y)) for x, y in polygon_points),
        ),
    )
    existing = build_texture_editor_selection_mask(document.width, document.height, document.selection)
    combined = _combine_selection_masks(existing, incoming, combine_mode=combine_mode)
    return dataclasses.replace(
        document,
        selection=_selection_from_mask(
            combined,
            feather_radius=max(0, int(document.selection.feather_radius)),
        ),
    )


def update_texture_editor_selection_settings(
    document: TextureEditorDocument,
    *,
    inverted: Optional[bool] = None,
    feather_radius: Optional[int] = None,
) -> TextureEditorDocument:
    selection = document.selection
    return dataclasses.replace(
        document,
        selection=dataclasses.replace(
            selection,
            inverted=selection.inverted if inverted is None else bool(inverted),
            feather_radius=max(0, int(selection.feather_radius if feather_radius is None else feather_radius)),
        ),
    )


def select_all_texture_editor(document: TextureEditorDocument) -> TextureEditorDocument:
    return dataclasses.replace(
        document,
        selection=TextureEditorSelection(
            mode="rect",
            rect=(0, 0, int(document.width), int(document.height)),
            inverted=False,
            feather_radius=max(0, int(document.selection.feather_radius)),
        ),
    )


def grow_texture_editor_selection(
    document: TextureEditorDocument,
    pixels: int,
) -> TextureEditorDocument:
    amount = max(0, int(pixels))
    mask = build_texture_editor_selection_mask(document.width, document.height, document.selection)
    if mask is None or amount <= 0:
        return document
    kernel_size = max(1, (amount * 2) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    grown = cv2.dilate(mask, kernel, iterations=1)
    return dataclasses.replace(
        document,
        selection=_selection_from_mask(
            grown,
            feather_radius=max(0, int(document.selection.feather_radius)),
        ),
    )


def shrink_texture_editor_selection(
    document: TextureEditorDocument,
    pixels: int,
) -> TextureEditorDocument:
    amount = max(0, int(pixels))
    mask = build_texture_editor_selection_mask(document.width, document.height, document.selection)
    if mask is None or amount <= 0:
        return document
    kernel_size = max(1, (amount * 2) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    shrunk = cv2.erode(mask, kernel, iterations=1)
    return dataclasses.replace(
        document,
        selection=_selection_from_mask(
            shrunk,
            feather_radius=max(0, int(document.selection.feather_radius)),
        ),
    )


def snap_lasso_points_to_edges(
    rgba_image: np.ndarray,
    polygon_points: Sequence[Tuple[float, float]],
    *,
    search_radius: int = 10,
    edge_sensitivity: int = 55,
) -> List[Tuple[float, float]]:
    if len(polygon_points) < 3:
        return [(float(x), float(y)) for x, y in polygon_points]
    radius = max(1, int(search_radius))
    sensitivity = max(1, min(100, int(edge_sensitivity)))
    rgba = np.asarray(rgba_image, dtype=np.uint8)
    gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2, sigmaY=1.2)
    high = max(40, min(220, int(round(210 - (sensitivity * 1.4)))))
    low = max(10, int(round(high * 0.45)))
    edges = cv2.Canny(blurred, low, high)
    snapped: List[Tuple[float, float]] = []
    height, width = edges.shape[:2]
    for x, y in polygon_points:
        px = max(0, min(width - 1, int(round(float(x)))))
        py = max(0, min(height - 1, int(round(float(y)))))
        x0 = max(0, px - radius)
        y0 = max(0, py - radius)
        x1 = min(width, px + radius + 1)
        y1 = min(height, py + radius + 1)
        patch = edges[y0:y1, x0:x1]
        if patch.size == 0 or not np.any(patch):
            snapped.append((float(px), float(py)))
            continue
        ys, xs = np.where(patch > 0)
        best_x = px
        best_y = py
        best_dist = None
        for local_x, local_y in zip(xs, ys):
            candidate_x = x0 + int(local_x)
            candidate_y = y0 + int(local_y)
            dist = ((candidate_x - px) ** 2) + ((candidate_y - py) ** 2)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_x = candidate_x
                best_y = candidate_y
        snapped.append((float(best_x), float(best_y)))
    deduped: List[Tuple[float, float]] = []
    for point in snapped:
        if not deduped or deduped[-1] != point:
            deduped.append(point)
    return deduped


@lru_cache(maxsize=1024)
def _build_brush_stamp(
    size: float,
    hardness: int,
    strength_percent: int,
    tip: str = "round",
    pattern: str = "solid",
    roundness: int = 100,
    angle_degrees: int = 0,
) -> np.ndarray:
    radius = max(0.5, float(size) / 2.0)
    diameter = max(1, int(math.ceil(radius * 2.0 + 2.0)))
    yy, xx = np.mgrid[0:diameter, 0:diameter].astype(np.float32)
    center = (diameter - 1) / 2.0
    dx = xx - center
    dy = yy - center
    radians = math.radians(float(angle_degrees))
    cos_value = math.cos(radians)
    sin_value = math.sin(radians)
    rotated_x = (dx * cos_value) + (dy * sin_value)
    rotated_y = (-dx * sin_value) + (dy * cos_value)
    roundness_ratio = max(0.15, min(1.0, float(roundness) / 100.0))
    scaled_x = rotated_x / max(roundness_ratio, 1e-6)
    scaled_y = rotated_y
    tip_key = (tip or "round").strip().lower()
    if tip_key == "square":
        distances = np.maximum(np.abs(scaled_x), np.abs(scaled_y))
    elif tip_key == "diamond":
        distances = (np.abs(scaled_x) + np.abs(scaled_y)) / math.sqrt(2.0)
    elif tip_key == "flat":
        distances = np.sqrt(((scaled_x / 1.5) ** 2) + ((scaled_y / 0.75) ** 2))
    else:
        distances = np.sqrt((scaled_x ** 2) + (scaled_y ** 2))
    outer = radius
    inner = outer * max(0.0, min(1.0, hardness / 100.0))
    stamp = np.zeros((diameter, diameter), dtype=np.float32)
    stamp[distances <= inner] = 1.0
    soft_mask = (distances > inner) & (distances <= outer)
    if np.any(soft_mask):
        stamp[soft_mask] = 1.0 - ((distances[soft_mask] - inner) / max(outer - inner, 1e-6))
    pattern_key = (pattern or "solid").strip().lower()
    if pattern_key != "solid":
        grid_y, grid_x = np.indices((diameter, diameter), dtype=np.float32)
        noise = np.sin((grid_x + 0.73) * 12.9898 + (grid_y + 1.41) * 78.233) * 43758.5453
        noise = noise - np.floor(noise)
        if pattern_key == "speckle":
            modulation = (noise > 0.58).astype(np.float32)
        elif pattern_key == "hatch":
            stripes = np.mod(grid_x + grid_y, 6.0)
            modulation = np.where(stripes < 2.0, 1.0, 0.22).astype(np.float32)
        elif pattern_key == "crosshatch":
            stripes_a = np.mod(grid_x + grid_y, 6.0)
            stripes_b = np.mod(grid_x - grid_y + (diameter * 2.0), 6.0)
            modulation = np.where((stripes_a < 2.0) | (stripes_b < 2.0), 1.0, 0.18).astype(np.float32)
        elif pattern_key == "grain":
            modulation = (0.42 + (noise * 0.58)).astype(np.float32)
        else:
            modulation = np.ones_like(stamp, dtype=np.float32)
        stamp *= modulation
    stamp *= max(0.0, min(1.0, strength_percent / 100.0))
    return np.clip(stamp, 0.0, 1.0)


def _interpolate_stroke(
    points: Sequence[Tuple[int, int]],
    spacing: int,
    *,
    smoothing: int = 0,
) -> List[Tuple[int, int]]:
    if not points:
        return []
    smoothed = _smooth_stroke_points(points, smoothing)
    if len(smoothed) == 1:
        return [(int(round(smoothed[0][0])), int(round(smoothed[0][1])))]
    output: List[Tuple[int, int]] = []
    step = max(1.0, float(spacing))
    for start, end in zip(smoothed[:-1], smoothed[1:]):
        x0, y0 = start
        x1, y1 = end
        dx = float(x1 - x0)
        dy = float(y1 - y0)
        distance = math.hypot(dx, dy)
        steps = max(1, int(math.ceil(distance / step)))
        for index in range(steps):
            t = index / steps
            output.append((int(round(x0 + dx * t)), int(round(y0 + dy * t))))
    output.append((int(round(smoothed[-1][0])), int(round(smoothed[-1][1]))))
    deduped: List[Tuple[int, int]] = []
    for point in output:
        if not deduped or deduped[-1] != point:
            deduped.append(point)
    return deduped


def _clip_stamp_region(
    point: Tuple[int, int],
    stamp: np.ndarray,
    width: int,
    height: int,
) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
    stamp_h, stamp_w = stamp.shape[:2]
    half_w = stamp_w // 2
    half_h = stamp_h // 2
    x0 = point[0] - half_w
    y0 = point[1] - half_h
    x1 = x0 + stamp_w
    y1 = y0 + stamp_h
    tx0 = max(0, x0)
    ty0 = max(0, y0)
    tx1 = min(width, x1)
    ty1 = min(height, y1)
    if tx1 <= tx0 or ty1 <= ty0:
        return None
    sx0 = tx0 - x0
    sy0 = ty0 - y0
    sx1 = sx0 + (tx1 - tx0)
    sy1 = sy0 + (ty1 - ty0)
    return (tx0, ty0, tx1, ty1), (sx0, sy0, sx1, sy1)


def _blend_constant_color(
    region: np.ndarray,
    stamp_alpha: np.ndarray,
    rgb: Tuple[int, int, int],
    *,
    mode: str = "normal",
) -> np.ndarray:
    dst = region.astype(np.float32) / 255.0
    src_alpha = stamp_alpha[..., None]
    src_rgb = np.zeros_like(dst[..., :3])
    src_rgb[..., 0] = rgb[0] / 255.0
    src_rgb[..., 1] = rgb[1] / 255.0
    src_rgb[..., 2] = rgb[2] / 255.0
    mode_key = (mode or "normal").strip().lower()
    if mode_key == "multiply":
        paint_rgb = dst[..., :3] * src_rgb
    elif mode_key == "screen":
        paint_rgb = 1.0 - ((1.0 - dst[..., :3]) * (1.0 - src_rgb))
    elif mode_key == "overlay":
        paint_rgb = np.where(
            dst[..., :3] <= 0.5,
            2.0 * dst[..., :3] * src_rgb,
            1.0 - (2.0 * (1.0 - dst[..., :3]) * (1.0 - src_rgb)),
        )
    else:
        paint_rgb = src_rgb
    dst_alpha = dst[..., 3:4]
    out_alpha = src_alpha + dst_alpha * (1.0 - src_alpha)
    safe_alpha = np.where(out_alpha > 1e-6, out_alpha, 1.0)
    out_rgb = (paint_rgb * src_alpha + dst[..., :3] * dst_alpha * (1.0 - src_alpha)) / safe_alpha
    out = dst.copy()
    out[..., :3] = np.where(out_alpha > 1e-6, out_rgb, out[..., :3])
    out[..., 3:4] = out_alpha
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


def _blend_patch(
    region: np.ndarray,
    patch: np.ndarray,
    stamp_alpha: np.ndarray,
) -> np.ndarray:
    dst = region.astype(np.float32) / 255.0
    src = patch.astype(np.float32) / 255.0
    src_alpha = src[..., 3:4] * stamp_alpha[..., None]
    dst_alpha = dst[..., 3:4]
    out_alpha = src_alpha + dst_alpha * (1.0 - src_alpha)
    safe_alpha = np.where(out_alpha > 1e-6, out_alpha, 1.0)
    out_rgb = (src[..., :3] * src_alpha + dst[..., :3] * dst_alpha * (1.0 - src_alpha)) / safe_alpha
    out = dst.copy()
    out[..., :3] = np.where(out_alpha > 1e-6, out_rgb, out[..., :3])
    out[..., 3:4] = out_alpha
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


def _apply_smudge_patch(
    target_region: np.ndarray,
    source_patch: np.ndarray,
    stamp_alpha: np.ndarray,
    strength: float,
) -> np.ndarray:
    weight = np.clip(stamp_alpha * max(0.0, min(1.0, strength)), 0.0, 1.0)
    return _blend_patch(target_region, source_patch, weight)


def _apply_dodge_burn_region(
    region: np.ndarray,
    stamp_alpha: np.ndarray,
    *,
    exposure: float,
    mode: str,
) -> np.ndarray:
    rgb = region[..., :3].astype(np.float32)
    luma = (0.299 * rgb[..., 0]) + (0.587 * rgb[..., 1]) + (0.114 * rgb[..., 2])
    normalized_luma = np.clip(luma / 255.0, 0.0, 1.0)
    mode_key = (mode or "dodge_midtones").strip().lower()
    if "shadows" in mode_key:
        tonal_weight = np.clip(1.0 - normalized_luma, 0.0, 1.0)
    elif "highlights" in mode_key:
        tonal_weight = np.clip(normalized_luma, 0.0, 1.0)
    else:
        tonal_weight = 1.0 - np.abs((normalized_luma * 2.0) - 1.0)
    weight = np.clip(stamp_alpha[..., None] * tonal_weight[..., None] * max(0.0, min(1.0, exposure)), 0.0, 1.0)
    adjusted = region.astype(np.float32)
    if mode_key.startswith("burn"):
        adjusted[..., :3] = np.clip(adjusted[..., :3] * (1.0 - (weight * 0.85)), 0.0, 255.0)
    else:
        adjusted[..., :3] = np.clip(adjusted[..., :3] + ((255.0 - adjusted[..., :3]) * weight * 0.85), 0.0, 255.0)
    return np.clip(np.round(adjusted), 0.0, 255.0).astype(np.uint8)


def _blend_gradient_color(
    start_rgb: Tuple[int, int, int],
    end_rgb: Tuple[int, int, int],
    amount: np.ndarray,
) -> np.ndarray:
    start = np.asarray(start_rgb, dtype=np.float32)
    end = np.asarray(end_rgb, dtype=np.float32)
    return np.clip(
        np.round((start[None, None, :] * (1.0 - amount[..., None])) + (end[None, None, :] * amount[..., None])),
        0.0,
        255.0,
    ).astype(np.uint8)


def _match_rgb_luma(target_rgb: Tuple[int, int, int], source_rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    target_luma = max(1.0, 0.299 * target_rgb[0] + 0.587 * target_rgb[1] + 0.114 * target_rgb[2])
    source_luma = 0.299 * source_rgb[0] + 0.587 * source_rgb[1] + 0.114 * source_rgb[2]
    scale = source_luma / target_luma
    return tuple(max(0, min(255, int(round(channel * scale)))) for channel in target_rgb)


def _blend_rgb(
    original_rgb: Tuple[int, int, int],
    target_rgb: Tuple[int, int, int],
    weight: float,
) -> Tuple[int, int, int]:
    return tuple(
        max(0, min(255, int(round((orig * (1.0 - weight)) + (target * weight)))))
        for orig, target in zip(original_rgb, target_rgb)
    )


def apply_texture_editor_recolor(
    image: np.ndarray,
    settings: TextureEditorToolSettings,
    *,
    selection_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    rgba = np.asarray(image, dtype=np.uint8).copy()
    target_rgb = _parse_hex_rgb(settings.recolor_target_hex, "#C85A30")
    source_rgb = _parse_hex_rgb(settings.recolor_source_hex, "#808080")
    tolerance = max(0.0, float(settings.recolor_tolerance))
    strength = max(0.0, min(1.0, settings.recolor_strength / 100.0))
    selection_alpha = selection_mask.astype(np.float32) / 255.0 if selection_mask is not None else None
    flat = rgba.reshape(-1, 4)
    selection_flat = selection_alpha.reshape(-1) if selection_alpha is not None else None
    for index, (r, g, b, a) in enumerate(flat):
        if a == 0:
            continue
        if selection_flat is not None and selection_flat[index] <= 0.0:
            continue
        base_rgb = (int(r), int(g), int(b))
        replacement_rgb = target_rgb
        weight = strength
        if settings.recolor_mode == "replace_color":
            distance = math.sqrt((r - source_rgb[0]) ** 2 + (g - source_rgb[1]) ** 2 + (b - source_rgb[2]) ** 2)
            if tolerance <= 0.0 or distance > tolerance:
                continue
            falloff = 1.0 - (distance / tolerance) if tolerance > 0.0 else 1.0
            weight *= max(0.0, min(1.0, falloff))
        if selection_flat is not None:
            weight *= float(selection_flat[index])
        if settings.recolor_preserve_luminance:
            replacement_rgb = _match_rgb_luma(target_rgb, base_rgb)
        merged_rgb = _blend_rgb(base_rgb, replacement_rgb, weight)
        flat[index, 0] = merged_rgb[0]
        flat[index, 1] = merged_rgb[1]
        flat[index, 2] = merged_rgb[2]
    return rgba


def apply_texture_editor_stroke(
    document: TextureEditorDocument,
    layer_pixels: Dict[str, np.ndarray],
    tool_settings: TextureEditorToolSettings,
    points: Sequence[Tuple[int, int]],
    *,
    source_snapshot: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    if not document.active_layer_id or document.active_layer_id not in layer_pixels:
        return layer_pixels
    active_layer = next((layer for layer in document.layers if layer.layer_id == document.active_layer_id), None)
    if active_layer is None or active_layer.locked:
        return layer_pixels
    selection_mask = build_texture_editor_selection_mask(document.width, document.height, document.selection)
    active = layer_pixels[document.active_layer_id].copy()
    updated = dict(layer_pixels)
    updated[document.active_layer_id] = active
    layer_height, layer_width = active.shape[:2]
    effective_size = _effective_brush_size(tool_settings)
    spacing = max(1, int(round(effective_size * max(0.05, tool_settings.spacing / 100.0))))
    stroke_points = _interpolate_stroke(points, spacing, smoothing=max(0, int(getattr(tool_settings, "smoothing", 0))))
    if not stroke_points:
        return updated
    strength_percent = max(0, min(100, int(round(tool_settings.opacity * max(0.05, tool_settings.flow / 100.0)))))
    brush_pattern = getattr(tool_settings, "brush_pattern", "solid")
    if tool_settings.tool not in {"paint", "erase", "clone", "heal", "smudge", "dodge_burn"}:
        brush_pattern = "solid"
    stamp = _build_brush_stamp(
        effective_size,
        max(0, min(100, tool_settings.hardness)),
        strength_percent,
        getattr(tool_settings, "brush_tip", "round"),
        brush_pattern,
        max(10, min(100, int(getattr(tool_settings, "roundness", 100)))),
        int(getattr(tool_settings, "angle_degrees", 0)),
    )
    color_rgb = _parse_hex_rgb(tool_settings.color_hex)
    if tool_settings.tool in {"clone", "heal"} and source_snapshot is None:
        return updated

    def _clip_to_active_layer(global_point: Tuple[int, int]) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
        local_point = (
            int(global_point[0] - active_layer.offset_x),
            int(global_point[1] - active_layer.offset_y),
        )
        return _clip_stamp_region(local_point, stamp, layer_width, layer_height)

    def _apply_alpha_lock(region_before: np.ndarray, region_after: np.ndarray) -> np.ndarray:
        locked = _apply_channel_edit_locks(document, region_before, region_after)
        if active_layer.alpha_locked:
            locked = locked.copy()
            locked[..., 3] = region_before[..., 3]
        return locked

    if source_snapshot is not None:
        sample_snapshot = np.zeros_like(active)
        intersection = _layer_canvas_intersection(active_layer, active, document)
        if intersection is not None:
            dx0, dy0, dx1, dy1, sx0, sy0, sx1, sy1 = intersection
            sample_snapshot[sy0:sy1, sx0:sx1] = source_snapshot[dy0:dy1, dx0:dx1]
    else:
        sample_snapshot = active.copy()

    stroke_origin = stroke_points[0]
    clone_delta: Optional[Tuple[int, int]] = None
    if tool_settings.tool in {"clone", "heal"} and tool_settings.clone_source_point is not None:
        clone_delta = (
            int(tool_settings.clone_source_point[0] - stroke_origin[0]),
            int(tool_settings.clone_source_point[1] - stroke_origin[1]),
        )

    if tool_settings.tool in {"soften", "sharpen"}:
        coverage = np.zeros((layer_height, layer_width), dtype=np.float32)
        for point in stroke_points:
            clipped = _clip_to_active_layer(point)
            if clipped is None:
                continue
            (lx0, ly0, lx1, ly1), (sx0, sy0, sx1, sy1) = clipped
            stamp_alpha = stamp[sy0:sy1, sx0:sx1]
            if selection_mask is not None:
                gx0 = active_layer.offset_x + lx0
                gy0 = active_layer.offset_y + ly0
                gx1 = gx0 + (lx1 - lx0)
                gy1 = gy0 + (ly1 - ly0)
                stamp_alpha = stamp_alpha * (selection_mask[gy0:gy1, gx0:gx1].astype(np.float32) / 255.0)
            if not np.any(stamp_alpha):
                continue
            current = coverage[ly0:ly1, lx0:lx1]
            coverage[ly0:ly1, lx0:lx1] = np.maximum(current, stamp_alpha)
        if not np.any(coverage):
            return updated

        sample_rgba = sample_snapshot.copy()
        rgba = active.copy()
        rgb = sample_rgba[..., :3].astype(np.float32)
        strength_ratio = max(0.0, min(1.0, tool_settings.strength / 100.0))
        sigma = max(0.8, float(effective_size) / 28.0)
        if tool_settings.tool == "sharpen":
            mode = getattr(tool_settings, "sharpen_mode", "unsharp_mask")
            if mode == "high_pass":
                blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma * 1.6, sigmaY=sigma * 1.6)
                detail = rgb - blurred
                amount = 0.04 + (strength_ratio * 0.20)
                processed_rgb = np.clip(rgb + (detail * amount), 0.0, 255.0)
                blend_strength = 0.08 + (strength_ratio * 0.34)
            elif mode == "local_contrast":
                blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma * 2.2, sigmaY=sigma * 2.2)
                detail = rgb - blurred
                amount = 0.03 + (strength_ratio * 0.12)
                processed_rgb = np.clip(rgb + (detail * amount), 0.0, 255.0)
                blend_strength = 0.07 + (strength_ratio * 0.30)
            else:
                blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma, sigmaY=sigma)
                amount = 0.05 + (strength_ratio * 0.42)
                processed_rgb = np.clip(cv2.addWeighted(rgb, 1.0 + amount, blurred, -amount, 0.0), 0.0, 255.0)
                blend_strength = 0.06 + (strength_ratio * 0.34)
        else:
            mode = getattr(tool_settings, "soften_mode", "gaussian")
            if mode == "median":
                kernel = max(3, int(round(effective_size / 18.0)) * 2 + 1)
                processed_rgb = cv2.medianBlur(np.clip(np.round(rgb), 0, 255).astype(np.uint8), kernel).astype(np.float32)
            elif mode == "surface":
                diameter = max(3, int(round(effective_size / 12.0)) * 2 + 1)
                sigma_color = 12.0 + (strength_ratio * 36.0)
                sigma_space = 6.0 + (strength_ratio * 20.0)
                processed_rgb = cv2.bilateralFilter(
                    np.clip(np.round(rgb), 0, 255).astype(np.uint8),
                    diameter,
                    sigma_color,
                    sigma_space,
                ).astype(np.float32)
            else:
                processed_rgb = cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma, sigmaY=sigma)
            blend_strength = 0.03 + (strength_ratio * 0.22)

        processed_rgba = sample_rgba.copy()
        processed_rgba[..., :3] = np.clip(np.round(processed_rgb), 0, 255).astype(np.uint8)
        blended = _blend_patch(
            rgba,
            processed_rgba,
            np.clip(coverage * blend_strength, 0.0, 1.0),
        )
        updated[document.active_layer_id] = _apply_alpha_lock(rgba, blended)
        return updated

    for point_index, point in enumerate(stroke_points):
        clipped = _clip_to_active_layer(point)
        if clipped is None:
            continue
        (lx0, ly0, lx1, ly1), (sx0, sy0, sx1, sy1) = clipped
        stamp_alpha = stamp[sy0:sy1, sx0:sx1]
        if selection_mask is not None:
            gx0 = active_layer.offset_x + lx0
            gy0 = active_layer.offset_y + ly0
            gx1 = gx0 + (lx1 - lx0)
            gy1 = gy0 + (ly1 - ly0)
            stamp_alpha = stamp_alpha * (selection_mask[gy0:gy1, gx0:gx1].astype(np.float32) / 255.0)
        if not np.any(stamp_alpha):
            continue
        region = active[ly0:ly1, lx0:lx1]
        if tool_settings.tool == "paint":
            active[ly0:ly1, lx0:lx1] = _apply_alpha_lock(
                region,
                _blend_constant_color(
                    region,
                    stamp_alpha,
                    color_rgb,
                    mode=getattr(tool_settings, "paint_blend_mode", "normal"),
                ),
            )
        elif tool_settings.tool == "erase":
            region_copy = region.copy().astype(np.float32)
            alpha = region_copy[..., 3] / 255.0
            alpha *= (1.0 - stamp_alpha)
            region_copy[..., 3] = np.clip(np.round(alpha * 255.0), 0, 255)
            active[ly0:ly1, lx0:lx1] = _apply_alpha_lock(region, region_copy.astype(np.uint8))
        elif tool_settings.tool == "smudge":
            previous_point = stroke_origin if point_index == 0 else stroke_points[point_index - 1]
            previous_clip = _clip_to_active_layer(previous_point)
            if previous_clip is None:
                continue
            (px0, py0, px1, py1), _ = previous_clip
            width = min(lx1 - lx0, px1 - px0)
            height = min(ly1 - ly0, py1 - py0)
            if width <= 0 or height <= 0:
                continue
            source_patch = sample_snapshot[py0:py0 + height, px0:px0 + width].copy()
            target_region = active[ly0:ly0 + height, lx0:lx0 + width]
            stamp_region = stamp_alpha[:height, :width]
            smudge_strength = max(0.0, min(1.0, getattr(tool_settings, "smudge_strength", 45) / 100.0))
            active[ly0:ly0 + height, lx0:lx0 + width] = _apply_alpha_lock(
                target_region,
                _apply_smudge_patch(target_region, source_patch, stamp_region, smudge_strength),
            )
        elif tool_settings.tool == "dodge_burn":
            active[ly0:ly1, lx0:lx1] = _apply_alpha_lock(
                region,
                _apply_dodge_burn_region(
                    region,
                    stamp_alpha,
                    exposure=max(0.0, min(1.0, getattr(tool_settings, "dodge_burn_exposure", 20) / 100.0)),
                    mode=str(getattr(tool_settings, "dodge_burn_mode", "dodge_midtones")),
                ),
            )
        elif tool_settings.tool in {"clone", "heal"} and source_snapshot is not None and clone_delta is not None:
            if getattr(tool_settings, "clone_aligned", True):
                source_center = (point[0] + clone_delta[0], point[1] + clone_delta[1])
            else:
                source_center = (
                    int(tool_settings.clone_source_point[0]),
                    int(tool_settings.clone_source_point[1]),
                )
            if tool_settings.sample_visible_layers:
                source_clip = _clip_stamp_region(source_center, stamp, document.width, document.height)
                patch_source = source_snapshot
            else:
                local_source_center = (
                    int(source_center[0] - active_layer.offset_x),
                    int(source_center[1] - active_layer.offset_y),
                )
                source_clip = _clip_stamp_region(local_source_center, stamp, layer_width, layer_height)
                patch_source = sample_snapshot
            if source_clip is None:
                continue
            (px0, py0, px1, py1), _ = source_clip
            width = min(lx1 - lx0, px1 - px0)
            height = min(ly1 - ly0, py1 - py0)
            if width <= 0 or height <= 0:
                continue
            patch = patch_source[py0:py0 + height, px0:px0 + width].copy()
            target_region = active[ly0:ly0 + height, lx0:lx0 + width]
            stamp_region = stamp_alpha[:height, :width]
            if tool_settings.tool == "heal":
                patch_rgb = patch[..., :3].astype(np.float32)
                target_rgb = target_region[..., :3].astype(np.float32)
                patch[..., :3] = np.clip(np.round((patch_rgb * 0.7) + (target_rgb * 0.3)), 0, 255).astype(np.uint8)
            active[ly0:ly0 + height, lx0:lx0 + width] = _apply_alpha_lock(
                target_region,
                _blend_patch(target_region, patch, stamp_region),
            )
    return updated


def apply_texture_editor_fill(
    document: TextureEditorDocument,
    layer_pixels: Dict[str, np.ndarray],
    tool_settings: TextureEditorToolSettings,
    point: Tuple[int, int],
    *,
    source_snapshot: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    if not document.active_layer_id or document.active_layer_id not in layer_pixels:
        return layer_pixels
    active_layer = next((layer for layer in document.layers if layer.layer_id == document.active_layer_id), None)
    if active_layer is None or active_layer.locked:
        return layer_pixels
    if point[0] < 0 or point[1] < 0 or point[0] >= document.width or point[1] >= document.height:
        return layer_pixels
    selection_mask = build_texture_editor_selection_mask(document.width, document.height, document.selection)
    if selection_mask is not None and selection_mask[point[1], point[0]] <= 0:
        return layer_pixels

    active = layer_pixels[document.active_layer_id].copy()
    updated = dict(layer_pixels)
    updated[document.active_layer_id] = active
    intersection = _layer_canvas_intersection(active_layer, active, document)
    if intersection is None:
        return updated
    dx0, dy0, dx1, dy1, sx0, sy0, sx1, sy1 = intersection
    if not (dx0 <= point[0] < dx1 and dy0 <= point[1] < dy1):
        return updated

    color_rgb = _parse_hex_rgb(tool_settings.color_hex)
    strength = max(0.0, min(1.0, tool_settings.opacity / 100.0))
    if strength <= 0.0:
        return updated

    if source_snapshot is None:
        sample_canvas = np.zeros((document.height, document.width, 4), dtype=np.uint8)
        sample_canvas[dy0:dy1, dx0:dx1] = active[sy0:sy1, sx0:sx1]
    else:
        sample_canvas = np.asarray(source_snapshot, dtype=np.uint8)

    tolerance = max(0, min(255, int(getattr(tool_settings, "fill_tolerance", 24))))
    contiguous = bool(getattr(tool_settings, "fill_contiguous", True))
    sample_rgb = sample_canvas[..., :3]
    seed = sample_rgb[point[1], point[0]].astype(np.int16)

    if contiguous:
        flood_source = cv2.cvtColor(sample_rgb, cv2.COLOR_RGB2BGR).copy()
        flood_mask = np.zeros((document.height + 2, document.width + 2), dtype=np.uint8)
        lo = (tolerance, tolerance, tolerance)
        hi = (tolerance, tolerance, tolerance)
        cv2.floodFill(
            flood_source,
            flood_mask,
            (int(point[0]), int(point[1])),
            (0, 0, 0),
            loDiff=lo,
            upDiff=hi,
            flags=4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY,
        )
        fill_mask = flood_mask[1:-1, 1:-1].astype(np.float32) / 255.0
    else:
        difference = np.max(np.abs(sample_rgb.astype(np.int16) - seed[None, None, :]), axis=2)
        fill_mask = (difference <= tolerance).astype(np.float32)

    if selection_mask is not None:
        fill_mask *= selection_mask.astype(np.float32) / 255.0
    if not np.any(fill_mask > 0.0):
        return updated

    local_fill = fill_mask[dy0:dy1, dx0:dx1]
    if not np.any(local_fill > 0.0):
        return updated
    region = active[sy0:sy1, sx0:sx1]
    blended = _blend_constant_color(
        region,
        np.clip(local_fill * strength, 0.0, 1.0),
        color_rgb,
        mode=getattr(tool_settings, "paint_blend_mode", "normal"),
    )
    blended = _apply_channel_edit_locks(document, region, blended)
    if active_layer.alpha_locked:
        blended[..., 3] = region[..., 3]
    active[sy0:sy1, sx0:sx1] = blended
    updated[document.active_layer_id] = active
    return updated


def apply_texture_editor_gradient(
    document: TextureEditorDocument,
    layer_pixels: Dict[str, np.ndarray],
    tool_settings: TextureEditorToolSettings,
    start_point: Tuple[int, int],
    end_point: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    if not document.active_layer_id or document.active_layer_id not in layer_pixels:
        return layer_pixels
    active_layer = next((layer for layer in document.layers if layer.layer_id == document.active_layer_id), None)
    if active_layer is None or active_layer.locked:
        return layer_pixels
    active = layer_pixels[document.active_layer_id].copy()
    updated = dict(layer_pixels)
    updated[document.active_layer_id] = active
    intersection = _layer_canvas_intersection(active_layer, active, document)
    if intersection is None:
        return updated
    dx0, dy0, dx1, dy1, sx0, sy0, sx1, sy1 = intersection
    region = active[sy0:sy1, sx0:sx1]
    selection_mask = build_texture_editor_selection_mask(document.width, document.height, document.selection)
    if selection_mask is not None:
        local_selection = selection_mask[dy0:dy1, dx0:dx1].astype(np.float32) / 255.0
    else:
        local_selection = np.ones((dy1 - dy0, dx1 - dx0), dtype=np.float32)
    if not np.any(local_selection > 0.0):
        return updated
    start_x = float(start_point[0] - dx0)
    start_y = float(start_point[1] - dy0)
    end_x = float(end_point[0] - dx0)
    end_y = float(end_point[1] - dy0)
    yy, xx = np.mgrid[0:(dy1 - dy0), 0:(dx1 - dx0)].astype(np.float32)
    gradient_mode = str(getattr(tool_settings, "gradient_type", "linear") or "linear").strip().lower()
    if gradient_mode == "radial":
        radius = max(1.0, math.hypot(end_x - start_x, end_y - start_y))
        amount = np.clip(np.sqrt(((xx - start_x) ** 2) + ((yy - start_y) ** 2)) / radius, 0.0, 1.0)
    else:
        vector_x = end_x - start_x
        vector_y = end_y - start_y
        denom = max(1e-6, (vector_x * vector_x) + (vector_y * vector_y))
        amount = np.clip((((xx - start_x) * vector_x) + ((yy - start_y) * vector_y)) / denom, 0.0, 1.0)
    start_rgb = _parse_hex_rgb(tool_settings.color_hex, "#C85A30")
    end_rgb = _parse_hex_rgb(getattr(tool_settings, "secondary_color_hex", "#FFFFFF"), "#FFFFFF")
    gradient_rgba = np.zeros_like(region)
    gradient_rgba[..., :3] = _blend_gradient_color(start_rgb, end_rgb, amount)
    gradient_rgba[..., 3] = np.clip(np.round(local_selection * (max(0.0, min(1.0, tool_settings.opacity / 100.0)) * 255.0)), 0.0, 255.0).astype(np.uint8)
    blended = _blend_layer_region(
        region,
        gradient_rgba,
        opacity=100,
        mode=getattr(tool_settings, "paint_blend_mode", "normal"),
    )
    blended = _apply_channel_edit_locks(document, region, blended)
    if active_layer.alpha_locked:
        blended[..., 3] = region[..., 3]
    active[sy0:sy1, sx0:sx1] = blended
    updated[document.active_layer_id] = active
    return updated


def apply_texture_editor_patch(
    document: TextureEditorDocument,
    layer_pixels: Dict[str, np.ndarray],
    tool_settings: TextureEditorToolSettings,
    *,
    delta_x: int,
    delta_y: int,
    source_snapshot: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    if not document.active_layer_id or document.active_layer_id not in layer_pixels:
        return layer_pixels
    active_layer = next((layer for layer in document.layers if layer.layer_id == document.active_layer_id), None)
    if active_layer is None or active_layer.locked:
        return layer_pixels
    selection_mask = build_texture_editor_selection_mask(document.width, document.height, document.selection)
    if selection_mask is None or not np.any(selection_mask > 0):
        return layer_pixels
    ys, xs = np.where(selection_mask > 0)
    if xs.size == 0 or ys.size == 0:
        return layer_pixels
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    active = layer_pixels[document.active_layer_id].copy()
    updated = dict(layer_pixels)
    updated[document.active_layer_id] = active
    intersection = _layer_canvas_intersection(active_layer, active, document)
    if intersection is None:
        return updated
    dx0, dy0, dx1, dy1, sx0, sy0, sx1, sy1 = intersection
    if x1 <= dx0 or x0 >= dx1 or y1 <= dy0 or y0 >= dy1:
        return updated
    region_x0 = max(dx0, x0)
    region_y0 = max(dy0, y0)
    region_x1 = min(dx1, x1)
    region_y1 = min(dy1, y1)
    local_x0 = region_x0 - dx0 + sx0
    local_y0 = region_y0 - dy0 + sy0
    local_x1 = local_x0 + (region_x1 - region_x0)
    local_y1 = local_y0 + (region_y1 - region_y0)
    if source_snapshot is None:
        sample_canvas = flatten_texture_editor_layers(document, layer_pixels)
    else:
        sample_canvas = np.asarray(source_snapshot, dtype=np.uint8)
    source_x0 = max(0, min(document.width, region_x0 + int(delta_x)))
    source_y0 = max(0, min(document.height, region_y0 + int(delta_y)))
    source_x1 = max(source_x0, min(document.width, source_x0 + (region_x1 - region_x0)))
    source_y1 = max(source_y0, min(document.height, source_y0 + (region_y1 - region_y0)))
    width = min(region_x1 - region_x0, source_x1 - source_x0)
    height = min(region_y1 - region_y0, source_y1 - source_y0)
    if width <= 0 or height <= 0:
        return updated
    target_region = active[local_y0:local_y0 + height, local_x0:local_x0 + width]
    source_patch = sample_canvas[source_y0:source_y0 + height, source_x0:source_x0 + width].copy()
    local_mask = selection_mask[region_y0:region_y0 + height, region_x0:region_x0 + width].astype(np.float32) / 255.0
    blend_strength = max(0.0, min(1.0, getattr(tool_settings, "patch_blend", 70) / 100.0))
    blended = _blend_patch(target_region, source_patch, np.clip(local_mask * blend_strength, 0.0, 1.0))
    blended = _apply_channel_edit_locks(document, target_region, blended)
    if active_layer.alpha_locked:
        blended[..., 3] = target_region[..., 3]
    active[local_y0:local_y0 + height, local_x0:local_x0 + width] = blended
    updated[document.active_layer_id] = active
    return updated


def capture_texture_editor_snapshot(
    document: TextureEditorDocument,
    layer_pixels: Dict[str, np.ndarray],
    label: str,
) -> Dict[str, object]:
    layer_blobs: Dict[str, bytes] = {}
    for layer_id, pixels in layer_pixels.items():
        encoded = cv2.imencode(".png", cv2.cvtColor(np.asarray(pixels, dtype=np.uint8), cv2.COLOR_RGBA2BGRA))[1]
        layer_blobs[layer_id] = bytes(encoded)
    return {
        "entry": TextureEditorHistoryEntry(label=label, timestamp=time.time()),
        "document": dataclasses.replace(document),
        "layer_blobs": layer_blobs,
    }


def restore_texture_editor_snapshot(snapshot: Dict[str, object]) -> Tuple[TextureEditorDocument, Dict[str, np.ndarray], TextureEditorHistoryEntry]:
    document = dataclasses.replace(snapshot["document"])  # type: ignore[arg-type]
    entry = snapshot["entry"]  # type: ignore[assignment]
    layer_pixels: Dict[str, np.ndarray] = {}
    for layer_id, blob in (snapshot.get("layer_blobs") or {}).items():
        decoded = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            continue
        if decoded.ndim == 2:
            decoded = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGRA)
        elif decoded.shape[2] == 3:
            decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2BGRA)
        rgba = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGBA)
        layer_pixels[str(layer_id)] = np.asarray(rgba, dtype=np.uint8).copy()
    return document, layer_pixels, entry


def extract_texture_editor_selection(
    document: TextureEditorDocument,
    layer_pixels: Dict[str, np.ndarray],
    layer_id: str,
) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    pixels = layer_pixels.get(layer_id)
    layer = next((candidate for candidate in document.layers if candidate.layer_id == layer_id), None)
    if pixels is None or layer is None:
        return None
    selection_mask = build_texture_editor_selection_mask(document.width, document.height, document.selection)
    if selection_mask is None:
        return None
    intersection = _layer_canvas_intersection(layer, pixels, document)
    if intersection is None:
        return None
    dx0, dy0, dx1, dy1, sx0, sy0, sx1, sy1 = intersection
    layer_selection = selection_mask[dy0:dy1, dx0:dx1]
    if not np.any(layer_selection):
        return None
    ys, xs = np.where(layer_selection > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    min_x = int(xs.min())
    min_y = int(ys.min())
    max_x = int(xs.max()) + 1
    max_y = int(ys.max()) + 1
    local_pixels = pixels[sy0 + min_y:sy0 + max_y, sx0 + min_x:sx0 + max_x].copy()
    local_alpha = np.clip(layer_selection[min_y:max_y, min_x:max_x].astype(np.float32) / 255.0, 0.0, 1.0)[..., None]
    extracted = np.clip(np.round(local_pixels.astype(np.float32) * local_alpha), 0, 255).astype(np.uint8)
    return extracted, (dx0 + min_x, dy0 + min_y, max_x - min_x, max_y - min_y)


def add_texture_editor_layer(
    document: TextureEditorDocument,
    layer_pixels: Dict[str, np.ndarray],
    *,
    name: str = "New Layer",
    initial_pixels: Optional[np.ndarray] = None,
    offset_x: int = 0,
    offset_y: int = 0,
    blend_mode: str = "normal",
) -> Tuple[TextureEditorDocument, Dict[str, np.ndarray], str]:
    new_id = _new_layer_id()
    new_layer = TextureEditorLayer(
        layer_id=new_id,
        name=name,
        relative_png_path=f"layers/{_safe_slug(name)}.png",
        visible=True,
        opacity=100,
        blend_mode=blend_mode,
        offset_x=int(offset_x),
        offset_y=int(offset_y),
        revision=0,
        thumbnail_cache_key=uuid.uuid4().hex,
    )
    layers = list(document.layers)
    layers.append(new_layer)
    new_pixels = dict(layer_pixels)
    if initial_pixels is None:
        new_pixels[new_id] = np.zeros((document.height, document.width, 4), dtype=np.uint8)
    else:
        new_pixels[new_id] = np.asarray(initial_pixels, dtype=np.uint8).copy()
    return dataclasses.replace(document, layers=tuple(layers), active_layer_id=new_id), new_pixels, new_id


def duplicate_texture_editor_layer(
    document: TextureEditorDocument,
    layer_pixels: Dict[str, np.ndarray],
    layer_id: str,
) -> Tuple[TextureEditorDocument, Dict[str, np.ndarray], Optional[str]]:
    source_layer = next((layer for layer in document.layers if layer.layer_id == layer_id), None)
    if source_layer is None or layer_id not in layer_pixels:
        return document, layer_pixels, None
    new_id = _new_layer_id()
    duplicated = dataclasses.replace(
        source_layer,
        layer_id=new_id,
        name=f"{source_layer.name} Copy",
        revision=int(source_layer.revision) + 1,
        thumbnail_cache_key=uuid.uuid4().hex,
    )
    layers = list(document.layers)
    insert_at = layers.index(source_layer) + 1
    layers.insert(insert_at, duplicated)
    new_pixels = dict(layer_pixels)
    new_pixels[new_id] = layer_pixels[layer_id].copy()
    if source_layer.mask_layer_id and source_layer.mask_layer_id in layer_pixels:
        duplicated_mask_id = _new_layer_id()
        duplicated = dataclasses.replace(
            duplicated,
            mask_layer_id=duplicated_mask_id,
            thumbnail_cache_key=uuid.uuid4().hex,
        )
        layers[insert_at] = duplicated
        new_pixels[duplicated_mask_id] = layer_pixels[source_layer.mask_layer_id].copy()
    return dataclasses.replace(document, layers=tuple(layers), active_layer_id=new_id), new_pixels, new_id


def remove_texture_editor_layer(
    document: TextureEditorDocument,
    layer_pixels: Dict[str, np.ndarray],
    layer_id: str,
) -> Tuple[TextureEditorDocument, Dict[str, np.ndarray]]:
    if len(document.layers) <= 1:
        return document, layer_pixels
    layers = [layer for layer in document.layers if layer.layer_id != layer_id]
    if len(layers) == len(document.layers):
        return document, layer_pixels
    new_pixels = dict(layer_pixels)
    new_pixels.pop(layer_id, None)
    active_layer_id = document.active_layer_id
    if active_layer_id == layer_id:
        active_layer_id = layers[-1].layer_id
    return dataclasses.replace(document, layers=tuple(layers), active_layer_id=active_layer_id), new_pixels


def merge_texture_editor_layer_down(
    document: TextureEditorDocument,
    layer_pixels: Dict[str, np.ndarray],
    layer_id: str,
) -> Tuple[TextureEditorDocument, Dict[str, np.ndarray]]:
    layers = list(document.layers)
    current_index = next((index for index, layer in enumerate(layers) if layer.layer_id == layer_id), -1)
    if current_index <= 0:
        return document, layer_pixels
    top_layer = layers[current_index]
    bottom_layer = layers[current_index - 1]
    top_pixels = layer_pixels.get(top_layer.layer_id)
    bottom_pixels = layer_pixels.get(bottom_layer.layer_id)
    if top_pixels is None or bottom_pixels is None:
        return document, layer_pixels
    merge_document = dataclasses.replace(
        document,
        layers=(bottom_layer, top_layer),
        active_layer_id=bottom_layer.layer_id,
    )
    merged_pixels = flatten_texture_editor_layers(
        merge_document,
        {
            bottom_layer.layer_id: bottom_pixels,
            top_layer.layer_id: top_pixels,
        },
    )
    new_pixels = dict(layer_pixels)
    new_pixels[bottom_layer.layer_id] = merged_pixels
    new_pixels.pop(top_layer.layer_id, None)
    layers[current_index - 1] = dataclasses.replace(
        bottom_layer,
        offset_x=0,
        offset_y=0,
        opacity=100,
        blend_mode="normal",
        revision=int(bottom_layer.revision) + 1,
        thumbnail_cache_key=uuid.uuid4().hex,
    )
    del layers[current_index]
    return dataclasses.replace(document, layers=tuple(layers), active_layer_id=bottom_layer.layer_id), new_pixels


def reorder_texture_editor_layer(
    document: TextureEditorDocument,
    layer_id: str,
    *,
    direction: int,
) -> TextureEditorDocument:
    layers = list(document.layers)
    index = next((pos for pos, layer in enumerate(layers) if layer.layer_id == layer_id), -1)
    if index < 0:
        return document
    target = index + int(direction)
    if target < 0 or target >= len(layers):
        return document
    layers[index], layers[target] = layers[target], layers[index]
    return dataclasses.replace(document, layers=tuple(layers))


def update_texture_editor_layer(
    document: TextureEditorDocument,
    layer_id: str,
    *,
    name: Optional[str] = None,
    visible: Optional[bool] = None,
    opacity: Optional[int] = None,
    blend_mode: Optional[str] = None,
    offset_x: Optional[int] = None,
    offset_y: Optional[int] = None,
    locked: Optional[bool] = None,
    alpha_locked: Optional[bool] = None,
) -> TextureEditorDocument:
    updated_layers: List[TextureEditorLayer] = []
    for layer in document.layers:
        if layer.layer_id != layer_id:
            updated_layers.append(layer)
            continue
        updated_layers.append(
            dataclasses.replace(
                layer,
                name=name if name is not None else layer.name,
                visible=visible if visible is not None else layer.visible,
                opacity=int(opacity) if opacity is not None else layer.opacity,
                blend_mode=blend_mode if blend_mode is not None else layer.blend_mode,
                offset_x=int(offset_x) if offset_x is not None else layer.offset_x,
                offset_y=int(offset_y) if offset_y is not None else layer.offset_y,
                locked=bool(locked) if locked is not None else layer.locked,
                alpha_locked=bool(alpha_locked) if alpha_locked is not None else layer.alpha_locked,
                revision=int(layer.revision) + 1,
                thumbnail_cache_key=uuid.uuid4().hex,
            )
        )
    return dataclasses.replace(document, layers=tuple(updated_layers))


def bump_texture_editor_layer_revision(
    document: TextureEditorDocument,
    layer_id: str,
) -> TextureEditorDocument:
    layer = next((candidate for candidate in document.layers if candidate.layer_id == layer_id), None)
    if layer is None:
        return document
    return update_texture_editor_layer(document, layer_id)


def move_texture_editor_layer(
    document: TextureEditorDocument,
    layer_id: str,
    *,
    dx: int,
    dy: int,
) -> TextureEditorDocument:
    if dx == 0 and dy == 0:
        return document
    layer = next((candidate for candidate in document.layers if candidate.layer_id == layer_id), None)
    if layer is None or layer.locked:
        return document
    return update_texture_editor_layer(
        document,
        layer_id,
        offset_x=int(layer.offset_x + dx),
        offset_y=int(layer.offset_y + dy),
    )


def create_texture_editor_layer_mask(
    document: TextureEditorDocument,
    layer_pixels: Dict[str, np.ndarray],
    layer_id: str,
) -> Tuple[TextureEditorDocument, Dict[str, np.ndarray], Optional[str]]:
    layer = next((candidate for candidate in document.layers if candidate.layer_id == layer_id), None)
    pixels = layer_pixels.get(layer_id)
    if layer is None or pixels is None:
        return document, layer_pixels, None
    if layer.mask_layer_id and layer.mask_layer_id in layer_pixels:
        updated_document = update_texture_editor_layer(document, layer_id, mask_enabled=True)
        return updated_document, dict(layer_pixels), layer.mask_layer_id
    mask_layer_id = _new_layer_id()
    mask_pixels = np.full_like(pixels, 255, dtype=np.uint8)
    mask_pixels[..., 0] = 255
    mask_pixels[..., 1] = 255
    mask_pixels[..., 2] = 255
    mask_pixels[..., 3] = 255
    updated_layers = []
    for candidate in document.layers:
        if candidate.layer_id != layer_id:
            updated_layers.append(candidate)
            continue
        updated_layers.append(
            dataclasses.replace(
                candidate,
                mask_layer_id=mask_layer_id,
                mask_enabled=True,
                revision=int(candidate.revision) + 1,
                thumbnail_cache_key=uuid.uuid4().hex,
            )
        )
    updated_document = dataclasses.replace(document, layers=tuple(updated_layers))
    new_pixels = dict(layer_pixels)
    new_pixels[mask_layer_id] = mask_pixels
    return updated_document, new_pixels, mask_layer_id


def set_texture_editor_layer_mask_enabled(
    document: TextureEditorDocument,
    layer_id: str,
    enabled: bool,
) -> TextureEditorDocument:
    layer = next((candidate for candidate in document.layers if candidate.layer_id == layer_id), None)
    if layer is None:
        return document
    updated_layers = [
        dataclasses.replace(
            candidate,
            mask_enabled=bool(enabled) if candidate.layer_id == layer_id else candidate.mask_enabled,
            revision=(candidate.revision + 1) if candidate.layer_id == layer_id else candidate.revision,
            thumbnail_cache_key=uuid.uuid4().hex if candidate.layer_id == layer_id else candidate.thumbnail_cache_key,
        )
        for candidate in document.layers
    ]
    return dataclasses.replace(document, layers=tuple(updated_layers))


def invert_texture_editor_layer_mask(
    document: TextureEditorDocument,
    layer_pixels: Dict[str, np.ndarray],
    layer_id: str,
) -> Dict[str, np.ndarray]:
    layer = next((candidate for candidate in document.layers if candidate.layer_id == layer_id), None)
    if layer is None or not layer.mask_layer_id or layer.mask_layer_id not in layer_pixels:
        return layer_pixels
    new_pixels = dict(layer_pixels)
    mask_pixels = new_pixels[layer.mask_layer_id].copy()
    mask_pixels[..., :3] = 255 - mask_pixels[..., :3]
    mask_pixels[..., 3] = 255 - mask_pixels[..., 3]
    new_pixels[layer.mask_layer_id] = mask_pixels
    return new_pixels


def delete_texture_editor_layer_mask(
    document: TextureEditorDocument,
    layer_pixels: Dict[str, np.ndarray],
    layer_id: str,
) -> Tuple[TextureEditorDocument, Dict[str, np.ndarray]]:
    layer = next((candidate for candidate in document.layers if candidate.layer_id == layer_id), None)
    if layer is None or not layer.mask_layer_id:
        return document, layer_pixels
    mask_layer_id = layer.mask_layer_id
    updated_layers = [
        dataclasses.replace(
            candidate,
            mask_layer_id="" if candidate.layer_id == layer_id else candidate.mask_layer_id,
            mask_enabled=False if candidate.layer_id == layer_id else candidate.mask_enabled,
            revision=(candidate.revision + 1) if candidate.layer_id == layer_id else candidate.revision,
            thumbnail_cache_key=uuid.uuid4().hex if candidate.layer_id == layer_id else candidate.thumbnail_cache_key,
        )
        for candidate in document.layers
    ]
    new_pixels = dict(layer_pixels)
    new_pixels.pop(mask_layer_id, None)
    return dataclasses.replace(document, layers=tuple(updated_layers)), new_pixels


def add_texture_editor_adjustment_layer(
    document: TextureEditorDocument,
    *,
    adjustment_type: str,
    name: str,
    parameters: Optional[Dict[str, float]] = None,
) -> TextureEditorDocument:
    adjustment = TextureEditorAdjustmentLayer(
        layer_id=_new_layer_id(),
        name=name,
        adjustment_type=adjustment_type,
        parameters=dict(parameters or {}),
        revision=0,
    )
    return dataclasses.replace(
        document,
        adjustment_layers=tuple(list(document.adjustment_layers) + [adjustment]),
        composite_revision=int(document.composite_revision) + 1,
    )


def update_texture_editor_adjustment_layer(
    document: TextureEditorDocument,
    adjustment_layer_id: str,
    *,
    enabled: Optional[bool] = None,
    opacity: Optional[int] = None,
    parameters: Optional[Dict[str, float]] = None,
    mask_layer_id: Optional[str] = None,
    name: Optional[str] = None,
) -> TextureEditorDocument:
    updated: List[TextureEditorAdjustmentLayer] = []
    changed = False
    for layer in document.adjustment_layers:
        if layer.layer_id != adjustment_layer_id:
            updated.append(layer)
            continue
        changed = True
        next_params = dict(layer.parameters)
        if parameters is not None:
            next_params.update(parameters)
        updated.append(
            dataclasses.replace(
                layer,
                name=name if name is not None else layer.name,
                enabled=bool(enabled) if enabled is not None else layer.enabled,
                opacity=int(opacity) if opacity is not None else layer.opacity,
                parameters=next_params,
                mask_layer_id=mask_layer_id if mask_layer_id is not None else layer.mask_layer_id,
                revision=int(layer.revision) + 1,
            )
        )
    if not changed:
        return document
    return dataclasses.replace(
        document,
        adjustment_layers=tuple(updated),
        composite_revision=int(document.composite_revision) + 1,
    )


def remove_texture_editor_adjustment_layer(
    document: TextureEditorDocument,
    adjustment_layer_id: str,
) -> TextureEditorDocument:
    remaining = tuple(layer for layer in document.adjustment_layers if layer.layer_id != adjustment_layer_id)
    if len(remaining) == len(document.adjustment_layers):
        return document
    return dataclasses.replace(document, adjustment_layers=remaining, composite_revision=int(document.composite_revision) + 1)
