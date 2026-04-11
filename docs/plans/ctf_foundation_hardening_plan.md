# CTF Foundation Hardening Plan

## Summary
This plan moves CTF toward being a primary DDS/texture workflow tool by shifting more texture correctness and technical-art decisions into the app itself. The focus is on semantic-aware processing paths, built-in output profiles, authoritative safety rules in every mode, and explicit support boundaries for technical textures.

The main engineering goal is to replace the current one-size-fits-all `DDS -> RGBA8 PNG -> backend -> DDS` assumption with a processing model that treats visible color textures, technical scalar maps, packed masks, alpha-sensitive textures, and precision-sensitive DDS formats as different classes with different allowed paths.

## Implementation Changes

### 1. Make the planner authoritative
- Refactor the existing policy/planning layer in `core/upscale_profiles.py` and `core/pipeline.py` so every file gets a single explicit processing plan before any staging happens.
- Replace the current implicit late decisions with a plan object containing:
  - semantic type and subtype
  - confidence and evidence
  - intermediate kind
  - backend compatibility
  - output profile
  - fallback action
  - risk flags
- Make this plan the only source of truth for:
  - `Preview Policy`
  - preflight reporting
  - DDS staging
  - backend execution eligibility
  - DDS rebuild behavior
  - compare/research metadata

### 2. Replace the universal PNG assumption with explicit processing paths
- Introduce three execution paths:
  - `visible_color_png_path`
  - `technical_preserve_path`
  - `technical_high_precision_path`
- Default path selection:
  - `color`, `ui`, `emissive`, `impostor`: allowed to use image-path processing
  - `normal`: preserve by default unless a declared normal-safe path is selected
  - `roughness`, `height`, `displacement`, `bump`, `ao`, `specular`, generic scalar masks: preserve by default until a grayscale-safe path exists
  - `packed_mask`, `opacity_mask`, `channel_data alpha`: preserve by default
  - `FLOAT`, `SNORM`, vector/pivot/position/flow DDS: always preserve in this tranche
- Remove the assumption that all DDS staging uses `R8G8B8A8_UNORM -> png`; instead route through per-path staging helpers.
- Acceptance rule: no technical texture may silently fall back to the generic RGBA8 PNG path.

### 3. Add built-in output profiles
- Add a curated internal profile table keyed by semantic subtype plus source DDS format.
- Initial built-in profiles:
  - `color_default`
  - `color_cutout_alpha`
  - `ui_alpha`
  - `normal_bc5`
  - `scalar_bc4`
  - `packed_mask_preserve_layout`
  - `premultiplied_alpha_review_required`
  - `float_or_vector_preserve_only`
- Each profile must declare:
  - allowed intermediate kinds
  - preferred DDS format
  - colorspace policy
  - alpha policy
  - mip policy hints
  - whether preserve-only is required
- Defaults should prefer `match original` unless a safer semantic mapping is clearly known.

### 4. Expand the semantic model and DDS metadata model
- Extend the current DDS/runtime types in `models.py`.
- Add new core types:
  - `TextureProcessingPlan`
  - `TextureProcessingProfile`
  - `IntermediateKind`
  - `BackendCapabilityMatrix`
  - `TextureSemanticEvidence`
  - `AlphaPolicy`
- Extend DDS metadata beyond width/height/mips/format/has_alpha to track at least:
  - colorspace intent
  - alpha intent classification
  - precision sensitivity
  - packed-channel risk
  - whether source format is preserve-only
- Do not attempt full cubemap/array/volume authoring in this tranche; detect and preserve them explicitly if encountered.

### 5. Make backend capability explicit
- Add a backend capability matrix instead of assuming all enabled backends can process all allowed textures.
- Initial declared capability rules:
  - `chaiNNer`: only trusted for classes whose selected chain output/intermediate guarantees the required path; otherwise preserve and log why
  - direct NCNN: visible-color path only
  - direct ONNX: visible-color path only unless a future normal/scalar-safe mode is explicitly implemented
  - backend disabled: still apply the same preserve-or-process plan
- This removes the current weak point where `backend == none` or aggressive presets can still lead to unsafe rebuilds from PNG.

### 6. Upgrade alpha handling from hints to policy
- Replace the current warning-only premult logic with explicit alpha policy states:
  - `none`
  - `straight`
  - `cutout_coverage`
  - `channel_data`
  - `premultiplied`
- Apply deterministic behavior per policy:
  - `cutout_coverage`: coverage-preserving mip generation
  - `channel_data`: preserve alpha as data, never treat it as display transparency
  - `premultiplied`: preserve-only in this tranche unless a true premult-safe path is implemented
- Keep heuristic detection, but make the resulting policy visible and binding.

### 7. Upgrade user overrides from raw texconv tweaks to semantic overrides
- Extend `TextureRule` from only `action/format/size/mips` to also support:
  - semantic override
  - profile override
  - colorspace override
  - alpha policy override
  - intermediate override
- Rules remain overrides, not the primary source of truth.
- Built-in profiles and planner defaults must work without requiring expert end users to write rules.

### 8. Move QA earlier and make it richer
- Keep `Research` and `Compare` as validators, but make them reflect the plan, not only post-facto drift.
- Add to compare/research output:
  - chosen processing path
  - chosen output profile
  - backend compatibility decision
  - preserve reason if preserved
  - “lossy intermediate used” warning when applicable
  - per-channel compare summary for rebuilt DDS previews
- Add a workflow-level summary count for:
  - preserved due to unsupported technical path
  - preserved due to precision-sensitive source
  - preserved due to alpha policy
  - rebuilt with visible-color path

## Public Interfaces / Type Changes
- `TextureRule` gains optional fields for semantic/profile/colorspace/alpha/intermediate overrides.
- Add `TextureProcessingPlan` as the plan output consumed by pipeline execution and UI/reporting.
- Add `TextureProcessingProfile` and `BackendCapabilityMatrix` as internal registries, but design them as explicit typed structures rather than ad hoc dicts.
- `Preview Policy` rows and research detail rows must expose:
  - profile
  - intermediate kind
  - alpha policy
  - backend compatibility
  - preserve reason
- `DdsInfo` should be extended conservatively with new semantic/runtime flags without breaking existing callers.

## Test Plan
- BC1 opaque color texture: planned as visible-color path, sRGB-aware, no false alpha warning.
- BC1 cutout texture: planned as cutout profile with coverage-preserving mip behavior.
- BC5 normal map: not allowed to silently route through generic color PNG path; either explicit normal-safe path or preserve.
- BC4 roughness: preserve by default until scalar-safe path exists.
- BC4 displacement and bump: preserve by default with subtype-specific reason.
- Packed mask with alpha data: preserve by default and expose `channel_data` alpha policy.
- FLOAT/SNORM/vector DDS: always preserve with explicit preflight/policy explanation.
- Backend disabled: same safe preserve/process decisions as when a backend is selected.
- `chaiNNer`, direct NCNN, and direct ONNX all log capability-based allow/deny decisions consistently.
- Manual texture-rule overrides supersede defaults and remain visible in policy preview and research details.
- Compare/research reports show the selected profile/path and do not require the user to infer why a file was preserved or rebuilt.

## Assumptions And Defaults
- Optimize for “primary DDS tool” credibility, not just wrapper convenience.
- Safety beats aggressiveness: unsupported technical textures are preserved by default.
- In this tranche, only visible color-like textures are allowed on the generic 8-bit image path by default.
- Premultiplied alpha, float, SNORM, and vector-style DDS are preserve-only until a true high-fidelity path exists.
- `texconv` remains the DDS encoder for now; the foundation change is that CTF becomes responsible for selecting the correct semantic path before encoding.
- Deliver in two implementation passes:
  - Pass 1: planner/profile/capability refactor, preserve-authority in all modes, alpha-policy hardening, richer UI/reporting
  - Pass 2: technical high-precision path for grayscale/scalar data and any backend/path support beyond visible-color textures
