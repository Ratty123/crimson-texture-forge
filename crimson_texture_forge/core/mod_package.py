from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

from crimson_texture_forge.models import ModPackageInfo


def sanitize_mod_package_folder_name(name: str) -> str:
    sanitized = re.sub(r'[<>:"/\\\\|?*]+', "_", name).strip(" .")
    return sanitized or "Crimson Texture Forge Mod"


def resolve_mod_package_root(parent_root: Path, package_info: ModPackageInfo) -> Path:
    package_title = (package_info.title or "").strip() or "Crimson Texture Forge Mod"
    return parent_root / sanitize_mod_package_folder_name(package_title)


def write_mod_package_info(
    root: Path,
    package_info: ModPackageInfo,
    *,
    create_no_encrypt_file: bool = True,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    no_encrypt_path = root / ".no_encrypt"
    if create_no_encrypt_file:
        no_encrypt_path.touch()
    elif no_encrypt_path.exists():
        no_encrypt_path.unlink()
    payload = {"modinfo": dataclasses.asdict(package_info)}
    (root / "info.json").write_text(json.dumps(payload, indent=4), encoding="utf-8")
