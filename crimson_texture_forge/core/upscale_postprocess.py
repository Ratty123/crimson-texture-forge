from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

from crimson_texture_forge.constants import (
    UPSCALE_POST_CORRECTION_MATCH_HISTOGRAM,
    UPSCALE_POST_CORRECTION_MATCH_LEVELS,
    UPSCALE_POST_CORRECTION_MATCH_MEAN_LUMA,
    UPSCALE_POST_CORRECTION_NONE,
)

try:
    from PIL import Image, ImageStat
except Exception:  # pragma: no cover - optional runtime dependency
    Image = None  # type: ignore[assignment]
    ImageStat = None  # type: ignore[assignment]

_VISIBLE_TEXTURE_TYPES = frozenset({"color", "ui", "emissive", "impostor"})
_POST_CORRECTION_MODES = (
    UPSCALE_POST_CORRECTION_NONE,
    UPSCALE_POST_CORRECTION_MATCH_MEAN_LUMA,
    UPSCALE_POST_CORRECTION_MATCH_LEVELS,
    UPSCALE_POST_CORRECTION_MATCH_HISTOGRAM,
)


@dataclass(slots=True)
class PostUpscaleCorrectionResult:
    applied: bool
    mode: str
    detail: str


def is_post_upscale_correction_supported() -> bool:
    return Image is not None and ImageStat is not None


def post_upscale_correction_error_message() -> str:
    return "Pillow is required for post-upscale color correction."


def normalize_post_upscale_correction_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower() or UPSCALE_POST_CORRECTION_NONE
    if normalized not in _POST_CORRECTION_MODES:
        raise ValueError(f"Unsupported post-upscale correction mode: {mode}")
    return normalized


def describe_post_upscale_correction_mode(mode: str) -> str:
    normalized = normalize_post_upscale_correction_mode(mode)
    if normalized == UPSCALE_POST_CORRECTION_MATCH_MEAN_LUMA:
        return "Match Mean Luma"
    if normalized == UPSCALE_POST_CORRECTION_MATCH_LEVELS:
        return "Match Levels"
    if normalized == UPSCALE_POST_CORRECTION_MATCH_HISTOGRAM:
        return "Match Histogram"
    return "Off"


def should_apply_post_upscale_correction(texture_type: str) -> bool:
    return str(texture_type or "").strip().lower() in _VISIBLE_TEXTURE_TYPES


def apply_post_upscale_color_correction(
    source_path: Path,
    output_path: Path,
    mode: str,
) -> PostUpscaleCorrectionResult:
    normalized_mode = normalize_post_upscale_correction_mode(mode)
    if normalized_mode == UPSCALE_POST_CORRECTION_NONE:
        return PostUpscaleCorrectionResult(False, normalized_mode, "disabled")
    if not is_post_upscale_correction_supported():
        raise RuntimeError(post_upscale_correction_error_message())

    source_rgb, _source_alpha = _load_rgb_and_alpha(Path(source_path))
    output_rgb, output_alpha = _load_rgb_and_alpha(Path(output_path))
    source_y, _source_cb, _source_cr = _split_luma(source_rgb)
    output_y, output_cb, output_cr = _split_luma(output_rgb)

    source_mean = _luma_mean(source_y)
    output_mean = _luma_mean(output_y)
    source_low, source_high = _histogram_bounds(source_y.histogram())
    output_low, output_high = _histogram_bounds(output_y.histogram())

    if normalized_mode == UPSCALE_POST_CORRECTION_MATCH_MEAN_LUMA:
        corrected_y = _apply_mean_luma_match(source_y, output_y)
    elif normalized_mode == UPSCALE_POST_CORRECTION_MATCH_LEVELS:
        corrected_y = _apply_levels_match(source_y, output_y)
    else:
        corrected_y = _apply_histogram_match(source_y, output_y)

    corrected_mean = _luma_mean(corrected_y)
    corrected_low, corrected_high = _histogram_bounds(corrected_y.histogram())
    corrected_image = _merge_luma(corrected_y, output_cb, output_cr, output_alpha)
    corrected_image.save(Path(output_path), format="PNG")

    detail = (
        f"{describe_post_upscale_correction_mode(normalized_mode)} "
        f"(mean {output_mean:.1f}->{corrected_mean:.1f} vs source {source_mean:.1f}; "
        f"range {output_low}-{output_high}->{corrected_low}-{corrected_high} vs source {source_low}-{source_high})"
    )
    return PostUpscaleCorrectionResult(True, normalized_mode, detail)


def _require_pillow() -> None:
    if not is_post_upscale_correction_supported():
        raise RuntimeError(post_upscale_correction_error_message())


def _load_rgb_and_alpha(path: Path) -> Tuple[Any, Optional[Any]]:
    _require_pillow()
    assert Image is not None
    if not path.exists() or not path.is_file():
        raise ValueError(f"PNG file does not exist: {path}")
    with Image.open(path) as image:
        alpha = image.getchannel("A").copy() if "A" in image.getbands() else None
        rgb = image.convert("RGB")
    return rgb, alpha


def _split_luma(image: Any) -> Tuple[Any, Any, Any]:
    ycbcr = image.convert("YCbCr")
    return ycbcr.split()


def _merge_luma(luma_channel: Any, cb_channel: Any, cr_channel: Any, alpha_channel: Optional[Any]) -> Any:
    assert Image is not None
    corrected_rgb = Image.merge("YCbCr", (luma_channel, cb_channel, cr_channel)).convert("RGB")
    if alpha_channel is not None:
        corrected_rgb.putalpha(alpha_channel)
    return corrected_rgb


def _luma_mean(channel: Any) -> float:
    assert ImageStat is not None
    return float(ImageStat.Stat(channel).mean[0])


def _histogram_bounds(histogram: list[int], *, low_percentile: float = 0.01, high_percentile: float = 0.99) -> Tuple[int, int]:
    return _histogram_percentile(histogram, low_percentile), _histogram_percentile(histogram, high_percentile)


def _histogram_percentile(histogram: list[int], percentile: float) -> int:
    total = int(sum(histogram))
    if total <= 0:
        return 0
    target = max(0.0, min(1.0, float(percentile))) * float(total - 1)
    running = 0.0
    for index, count in enumerate(histogram):
        running += float(count)
        if running > target:
            return index
    return 255


def _apply_mean_luma_match(source_y: Any, output_y: Any) -> Any:
    delta = _luma_mean(source_y) - _luma_mean(output_y)
    lut = [_clamp_byte(value + delta) for value in range(256)]
    return output_y.point(lut)


def _apply_levels_match(source_y: Any, output_y: Any) -> Any:
    source_hist = source_y.histogram()
    output_hist = output_y.histogram()
    source_low, source_high = _histogram_bounds(source_hist)
    output_low, output_high = _histogram_bounds(output_hist)
    if source_high <= source_low or output_high <= output_low:
        return _apply_mean_luma_match(source_y, output_y)

    scale = float(source_high - source_low) / float(output_high - output_low)
    lut = [_clamp_byte(source_low + ((value - output_low) * scale)) for value in range(256)]
    return output_y.point(lut)


def _apply_histogram_match(source_y: Any, output_y: Any) -> Any:
    source_hist = source_y.histogram()
    output_hist = output_y.histogram()
    source_cdf = _build_cdf(source_hist)
    output_cdf = _build_cdf(output_hist)
    lut: list[int] = []
    source_index = 0
    for value in range(256):
        target = output_cdf[value]
        while source_index < 255 and source_cdf[source_index] < target:
            source_index += 1
        lut.append(source_index)
    return output_y.point(lut)


def _build_cdf(histogram: list[int]) -> list[float]:
    total = float(sum(histogram))
    if total <= 0.0:
        return [0.0] * 256
    cdf: list[float] = []
    running = 0.0
    for count in histogram:
        running += float(count)
        cdf.append(running / total)
    return cdf


def _clamp_byte(value: float) -> int:
    return max(0, min(255, int(round(float(value)))))
