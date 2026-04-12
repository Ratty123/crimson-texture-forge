from __future__ import annotations

import shlex
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from crimson_texture_forge.constants import (
    UPSCALE_BACKEND_REALESRGAN_NCNN,
    UPSCALE_POST_CORRECTION_NONE,
    UPSCALE_TEXTURE_PRESET_ALL,
    UPSCALE_TEXTURE_PRESET_BALANCED,
    UPSCALE_TEXTURE_PRESET_COLOR_UI,
    UPSCALE_TEXTURE_PRESET_COLOR_UI_EMISSIVE,
)
from crimson_texture_forge.core.common import raise_if_cancelled, run_process_with_cancellation
from crimson_texture_forge.core.upscale_postprocess import (
    apply_post_upscale_color_correction,
    build_source_match_plan_for_decision,
    describe_post_upscale_correction_mode,
)
from crimson_texture_forge.core.upscale_profiles import (
    build_ncnn_retry_tile_candidates,
    copy_mod_ready_loose_tree,
    describe_texture_preset,
)
from crimson_texture_forge.models import NormalizedConfig, TextureProcessingPlan


def resolve_ncnn_model_dir(ncnn_exe_path: Optional[Path], explicit_model_dir: Optional[Path]) -> Optional[Path]:
    if explicit_model_dir is not None:
        return explicit_model_dir
    if ncnn_exe_path is None:
        return None
    default_dir = ncnn_exe_path.parent / "models"
    if default_dir.exists() and default_dir.is_dir():
        return default_dir
    return None


def discover_realesrgan_ncnn_models(
    ncnn_exe_path: Optional[Path],
    model_dir: Optional[Path],
) -> List[Tuple[str, Path]]:
    resolved_dir = resolve_ncnn_model_dir(ncnn_exe_path, model_dir)
    if resolved_dir is None or not resolved_dir.exists() or not resolved_dir.is_dir():
        return []

    discovered: List[Tuple[str, Path]] = []
    for param_path in sorted(resolved_dir.glob("*.param")):
        if not param_path.is_file():
            continue
        bin_path = param_path.with_suffix(".bin")
        if not bin_path.exists():
            continue
        discovered.append((param_path.stem, resolved_dir))
    return discovered


def build_realesrgan_ncnn_command(
    ncnn_exe_path: Path,
    *,
    input_path: Path,
    output_path: Path,
    model_dir: Path,
    model_name: str,
    scale: int,
    tile_size: int,
    extra_args: Sequence[str] = (),
) -> List[str]:
    cmd = [
        str(ncnn_exe_path),
        "-i",
        str(input_path),
        "-o",
        str(output_path),
        "-m",
        str(model_dir),
        "-n",
        model_name,
        "-s",
        str(scale),
        "-t",
        str(tile_size),
        "-f",
        "png",
    ]
    if extra_args:
        cmd.extend(str(arg) for arg in extra_args if str(arg).strip())
    return cmd


def parse_realesrgan_ncnn_extra_args(raw_text: str) -> List[str]:
    text = str(raw_text or "").strip()
    if not text:
        return []
    try:
        return shlex.split(text, posix=True)
    except ValueError as exc:
        raise ValueError(f"Real-ESRGAN NCNN extra args are invalid: {exc}") from exc


def _run_single_ncnn_attempt(
    config: NormalizedConfig,
    *,
    input_root: Path,
    output_root: Path,
    tile_size: int,
    processing_plan: Sequence[TextureProcessingPlan] = (),
    on_log: Optional[Callable[[str], None]] = None,
    on_phase_progress: Optional[Callable[[int, int, str], None]] = None,
    on_current_file: Optional[Callable[[str], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> None:
    plan_entries = [entry for entry in processing_plan if entry.requires_png_processing]
    if not plan_entries:
        if on_log:
            on_log("No PNG files require Real-ESRGAN NCNN processing under the current plan; skipping backend stage.")
        if on_phase_progress:
            on_phase_progress(0, 0, "0 / 0 PNG files")
        return

    png_inputs = [input_root / entry.relative_path.with_suffix(".png") for entry in plan_entries]
    total = len(png_inputs)
    if total == 0:
        raise ValueError(
            f"No PNG files were found for Real-ESRGAN NCNN in {input_root}. "
            "Enable DDS staging first or populate PNG root with source PNG files."
        )

    if on_log:
        on_log(f"Real-ESRGAN NCNN executable: {config.ncnn_exe_path}")
        on_log(f"Real-ESRGAN NCNN model folder: {config.ncnn_model_dir}")
        on_log(f"Real-ESRGAN NCNN model: {config.ncnn_model_name}")
        on_log(f"Real-ESRGAN NCNN scale={config.ncnn_scale}, tile={tile_size}, preset={config.upscale_texture_preset}")
        if config.ncnn_extra_args.strip():
            on_log(f"Real-ESRGAN NCNN extra args={config.ncnn_extra_args}")
        on_log(
            f"Real-ESRGAN NCNN post correction={describe_post_upscale_correction_mode(config.upscale_post_correction_mode)}"
        )
        on_log(describe_texture_preset(config.upscale_texture_preset))

    if on_phase_progress:
        on_phase_progress(0, total, f"0 / {total} PNG files")

    assert config.ncnn_exe_path is not None
    assert config.ncnn_model_dir is not None
    parsed_extra_args = parse_realesrgan_ncnn_extra_args(config.ncnn_extra_args)

    plan_by_rel = {entry.relative_path.as_posix(): entry for entry in processing_plan}

    for index, input_png in enumerate(png_inputs, start=1):
        raise_if_cancelled(stop_event)
        if not input_png.exists() or not input_png.is_file():
            raise ValueError(f"Expected planner-selected PNG does not exist: {input_png}")
        rel_path = input_png.relative_to(input_root)
        rel_display = rel_path.as_posix()
        plan_entry = plan_by_rel.get(rel_path.with_suffix(".dds").as_posix()) or plan_by_rel.get(rel_display.replace(".png", ".dds"))
        if plan_entry is None:
            raise ValueError(f"Missing planner entry for NCNN input: {rel_display}")
        decision = plan_entry.decision
        correction_plan = build_source_match_plan_for_decision(
            config.upscale_post_correction_mode,
            decision,
            direct_backend_supported=True,
            planner_path_kind=plan_entry.path_kind,
            planner_profile_key=plan_entry.profile.key,
        )
        texture_type = decision.texture_type or "unknown"
        output_png = output_root / rel_path
        output_png.parent.mkdir(parents=True, exist_ok=True)
        if on_current_file:
            on_current_file(f"Upscale: {rel_display}")

        if plan_entry.action == "upscale_then_rebuild":
            cmd = build_realesrgan_ncnn_command(
                config.ncnn_exe_path,
                input_path=input_png,
                output_path=output_png,
                model_dir=config.ncnn_model_dir,
                model_name=config.ncnn_model_name,
                scale=config.ncnn_scale,
                tile_size=tile_size,
                extra_args=parsed_extra_args,
            )
            action = "DRYRUN" if config.dry_run else "UPSCALE"
            if on_log:
                on_log(f"[{index}/{total}] {action} {rel_display} [{texture_type}]")
            if not config.dry_run:
                return_code, stdout, stderr = run_process_with_cancellation(cmd, stop_event=stop_event)
                if return_code != 0:
                    detail = stderr.strip() or stdout.strip() or f"Real-ESRGAN NCNN failed with exit code {return_code}"
                    raise ValueError(f"Real-ESRGAN NCNN failed for {rel_display}: {detail}")
                if config.upscale_post_correction_mode != UPSCALE_POST_CORRECTION_NONE:
                    raise_if_cancelled(stop_event)
                    correction_result = apply_post_upscale_color_correction(
                        input_png,
                        output_png,
                        config.upscale_post_correction_mode,
                        correction_plan=correction_plan,
                    )
                    if on_log and correction_result.applied:
                        on_log(f"[{index}/{total}] CORRECT {rel_display} [{texture_type}] -> {correction_result.detail}")
                    elif on_log and correction_result.correction_action == "skip":
                        on_log(
                            f"[{index}/{total}] SKIP CORRECTION {rel_display} [{texture_type}] "
                            f"-> {correction_result.correction_reason}"
                        )
        else:
            raise ValueError(f"Planner selected unexpected NCNN action for {rel_display}: {plan_entry.action}")

        if on_phase_progress:
            on_phase_progress(index, total, f"{index} / {total} PNG files")


def run_realesrgan_ncnn_stage(
    config: NormalizedConfig,
    *,
    processing_plan: Sequence[TextureProcessingPlan] = (),
    on_log: Optional[Callable[[str], None]] = None,
    on_phase: Optional[Callable[[str, str, bool], None]] = None,
    on_phase_progress: Optional[Callable[[int, int, str], None]] = None,
    on_current_file: Optional[Callable[[str], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> None:
    if config.upscale_backend != UPSCALE_BACKEND_REALESRGAN_NCNN:
        return
    if config.ncnn_exe_path is None or config.ncnn_model_dir is None or not config.ncnn_model_name:
        raise ValueError("Real-ESRGAN NCNN is selected, but the executable, model folder, or model name is missing.")

    input_root = config.dds_staging_root if config.enable_dds_staging and config.dds_staging_root is not None else config.png_root
    if not input_root.exists() or not input_root.is_dir():
        raise ValueError(f"Real-ESRGAN NCNN input folder does not exist: {input_root}")

    retry_plan = build_ncnn_retry_tile_candidates(config.ncnn_tile_size, include_full_frame_fallback=False)
    candidate_tiles = (retry_plan.requested_tile_size, *retry_plan.candidate_tile_sizes)
    attempt_tiles = (
        candidate_tiles
        if config.retry_smaller_tile_on_failure and not config.dry_run
        else (retry_plan.requested_tile_size,)
    )

    last_error: Optional[Exception] = None
    total_attempts = len(attempt_tiles)

    if on_phase:
        on_phase("Upscaling", "Running Real-ESRGAN NCNN...", False)
    if on_log:
        on_log("Phase 1/2: running Real-ESRGAN NCNN.")
        on_log(f"Real-ESRGAN NCNN input folder: {input_root}")
        on_log(f"Real-ESRGAN NCNN retry tile candidates: {', '.join(str(tile) for tile in attempt_tiles)}")

    for attempt_index, tile_size in enumerate(attempt_tiles, start=1):
        attempt_output_root = Path(tempfile.mkdtemp(prefix="crimson_texture_forge_ncnn_"))
        attempt_succeeded = False
        try:
            if on_log and total_attempts > 1:
                on_log(f"NCNN attempt {attempt_index}/{total_attempts} using tile size {tile_size}.")
            _run_single_ncnn_attempt(
                config,
                input_root=input_root,
                output_root=attempt_output_root,
                tile_size=tile_size,
                processing_plan=processing_plan,
                on_log=on_log,
                on_phase_progress=on_phase_progress,
                on_current_file=on_current_file,
                stop_event=stop_event,
            )
            attempt_succeeded = True
            if not config.dry_run:
                if on_log:
                    on_log(f"Syncing Real-ESRGAN NCNN output back into PNG root: {config.png_root}")
                copy_mod_ready_loose_tree(
                    attempt_output_root,
                    config.png_root,
                    overwrite=True,
                    dry_run=False,
                    on_log=None,
                )
            if on_log:
                on_log("Real-ESRGAN NCNN completed successfully.")
            return
        except Exception as exc:
            last_error = exc
            if on_log:
                on_log(f"Real-ESRGAN NCNN attempt with tile {tile_size} failed: {exc}")
            if attempt_index < total_attempts:
                next_tile = attempt_tiles[attempt_index]
                if on_log:
                    on_log(f"Retrying Real-ESRGAN NCNN with smaller tile size {next_tile}.")
            else:
                raise
        finally:
            if attempt_output_root.exists():
                shutil.rmtree(attempt_output_root, ignore_errors=True)

    if last_error is not None:
        raise last_error
