# Crimson Texture Forge

Windows desktop tool for **Crimson Desert texture workflows** and supporting archive/text-search tasks.

Project changelog: [CHANGELOG.md](CHANGELOG.md)

Crimson Texture Forge is built for modders who want one place to:

- browse and extract files from `.pamt` / `.paz` archives
- bulk convert DDS to PNG with `texconv`
- optionally run bulk upscaling through `chaiNNer`
- rebuild final DDS output with controlled format, size, and mip settings
- compare original vs rebuilt DDS files
- search archive or loose text-like files such as `.xml`, `.json`, `.cfg`, and `.lua`

The app is intentionally focused on **read-only archive access** and **loose-file workflows**. It does **not** repack or write back into game archives.

## Highlights

- Read-only archive browser for Crimson Desert `.pamt` / `.paz`
- Archive extraction to normal folders with preserved structure
- Archive cache for faster repeated scans
- Text Search tab with syntax-colored preview, line numbers, local find controls, and export of matched files
- Support for searching many encrypted archive XML files through deterministic ChaCha20 decryption where supported
- Loose DDS scan/filter workflow
- Optional DDS-to-PNG conversion before processing
- Optional `chaiNNer` stage before DDS rebuild
- DDS rebuild with configurable format, size, and mip behavior
- Side-by-side compare view for original vs rebuilt DDS
- Local auto-save config beside the EXE
- Profile export/import and diagnostic bundle export

## Recent Performance Work

Archive refresh and cache build were heavily optimized.

On the current large test archive set used during development, the refresh path dropped from **minutes** to **seconds** after fixing the real bottlenecks in `.pamt` parsing and cache generation. Exact timings depend on hardware and storage, but this was a substantial improvement rather than a minor tweak.

## Screenshots

### Archive Browser

![Archive Browser](docs/screenshots/archive_browser.png)

### Workflow: DDS To PNG / DDS Rebuild

![Workflow PNG to DDS](docs/screenshots/workflow_png_to_dds.png)

### Workflow: Upscaling Running

![Workflow Upscaling Running](docs/screenshots/workflow_upscalingrunning.png)

### Compare View

![Workflow Compare](docs/screenshots/workflow_compare.png)

### Compare View: Alternate Example

![Workflow Compare 2](docs/screenshots/workflow_compare2.png)

### Text Search

![Text Search](docs/screenshots/TextSearcher.png)

## Quick Start

### Basic texture workflow

1. Run `CrimsonTextureForge-<version>-windows-portable.exe`.
2. In `Workflow > Setup`, click `Init Workspace`.
3. Configure or download `texconv.exe`.
4. Set `Original DDS root`, `PNG root`, and `Output root`.
5. Click `Scan`.
6. If you want PNG files first, enable `Convert DDS to PNG before processing`.
7. Click `Start`.
8. Review the results in `Compare`.

### Archive to workflow

1. Open `Archive Browser`.
2. Set or auto-detect `Package root`.
3. Click `Scan` or `Refresh`.
4. Filter the archive entries.
5. Use `DDS To Workflow` or extract selected DDS files to a normal folder.
6. Return to `Workflow`, click `Scan`, and run a small test first.

### Text search

1. Open `Text Search`.
2. Choose `Archive files` after scanning archives, or `Loose folder` for extracted content.
3. Enter a search string or regex.
4. Set extensions such as `.xml;.json;.cfg`.
5. Click `Search`.
6. Preview matches with syntax colors, line numbers, and highlighted results.
7. Export selected or all matched files while preserving folder structure.

## Main Tabs

### Workflow

Use this for DDS/PNG processing:

- loose DDS scanning and filtering
- optional DDS-to-PNG conversion
- optional `chaiNNer` run
- DDS rebuild through `texconv`
- compare original vs rebuilt DDS

### Archive Browser

Use this for read-only archive work:

- scan or refresh `.pamt` / `.paz`
- filter by package, folder, path, extension, role, and size
- preview supported assets
- extract selected files to normal folders
- send DDS files directly into the workflow

### Text Search

Use this as a supporting modding utility:

- search archive or loose text-like files
- decrypt supported encrypted XML entries when possible
- preview results in an editor-style view
- export matched files with folder structure preserved

### Settings

Global persistent preferences live here, including:

- theme
- auto-load archive browser on startup
- prefer cache on startup
- restore last active tab
- remember splitter sizes
- safety prompts for cleanup actions

## Optional chaiNNer Workflow

`chaiNNer` is optional and external.

Crimson Texture Forge can:

- launch `chaiNNer`
- inspect a `.chn` file
- report likely path mismatches
- warn about missing output

Crimson Texture Forge cannot:

- build your chain for you
- install your nodes or backends for you
- guarantee a chain is valid
- guarantee that GUI and CLI behavior will be identical

Before using `chaiNNer`:

1. Install `chaiNNer` separately.
2. Launch it directly at least once.
3. Install the backends your chain needs, such as:
   - PyTorch
   - NCNN
   - ONNX / ONNX Runtime
4. Create and test your own `.chn` chain first.
5. Make sure the chain reads the correct input type from the correct folder.

If your chain expects PNG input:

- enable `Convert DDS to PNG before processing`
- point the chain at `${staging_png_root}`, `${png_root}`, or another matching PNG folder

If your chain reads DDS directly:

- verify that in `chaiNNer` itself first
- do not enable DDS-to-PNG conversion unless the chain is meant to use PNG output

Supported override tokens:

- `${original_dds_root}`
- `${staging_png_root}`
- `${png_root}`
- `${output_root}`
- `${texconv_path}`

## DDS Output Behavior

Final DDS output is built with `texconv`.

You can:

- match the original DDS format
- choose a custom format
- keep original size, PNG size, or use a custom size
- match original mip count, generate a full mip chain, force a single mip, or use a custom mip count

You can also apply:

- include filters
- per-pattern texture rules

## Safety And Scope

What the app does:

- reads archive indexes and data
- extracts selected files to normal folders
- processes loose files in workspace folders
- writes loose PNG/DDS output to folders you choose

What the app does not do:

- repack `.pamt` / `.paz`
- modify game archives in place
- guarantee correct output for every texture type without user judgment

## Local Config, Cache, Profiles, Diagnostics

The app stores its local state beside the EXE.

Main local files/folders:

- `CrimsonTextureForge.cfg`
- `archive_cache`

The app also supports:

- profile export/import
- diagnostic bundle export

## Privacy And Network Behavior

Crimson Texture Forge does **not** include built-in telemetry, analytics, auto-update checks, or background network calls for normal offline use.

The app only makes direct network requests when you explicitly trigger download actions such as:

- `Download chaiNNer`
- `Download texconv`

Also note:

- opening external links from the app is user-initiated and handled by your default browser
- Windows, SmartScreen, certificate validation, or antivirus tools may still perform their own reputation/certificate checks around a newly built EXE

## Known Limitations

- Archive access is read-only.
- Archive preview for some unusual DDS cases is still best-effort.
- `chaiNNer` reliability depends on the chain and dependencies you provide.
- Some chain behavior cannot be fully inferred from a `.chn` file alone.
- Final DDS quality depends on source PNG quality, chosen rebuild settings, and texture type.

## Dependencies

### Python packages used by the app

- `PySide6`
- `PyInstaller`
- `lz4`
- `cryptography`

Build requirements are listed in `requirements-build.txt`.

### External tools used by the app

- `texconv.exe` from Microsoft DirectXTex
- `chaiNNer` for the optional upscaling stage

## Development

Example local setup:

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements-build.txt
.\.venv\Scripts\python crimson_texture_forge_app.py
```

One-file build:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_pyside6_app.ps1
```

## Credits And References

Crimson Texture Forge uses or depends on the following:

- `PySide6 / Qt` for the desktop UI
- `PyInstaller` for packaging
- `DirectXTex / texconv` by Microsoft for DDS conversion and preview generation
- `chaiNNer` for the optional external upscaling stage
- `lz4` for archive decompression support

Archive support and compatibility work were informed by community reverse-engineering and tool behavior around Crimson Desert packages, especially:

- [Lazorr](https://forums.nexusmods.com/profile/194233100-lazorr/) and [Crimson Desert Unpacker](https://www.nexusmods.com/crimsondesert/mods/62)
- [IzeDev](https://www.nexusmods.com/profile/IzeDev) and [Crimson Browser & Mod Manager](https://www.nexusmods.com/crimsondesert/mods/84)

See also:

- `LICENSE`
- `THIRD_PARTY_NOTICES.md`

## License

This repository is published under the `MIT` license. See `LICENSE`.
