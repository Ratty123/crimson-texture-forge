from __future__ import annotations

from PyInstaller.utils.hooks import collect_all


def _keep_entry(entry: object) -> bool:
    try:
        path_text = str(entry[0]).lower()
    except Exception:
        path_text = str(entry).lower()
    return "avif" not in path_text


datas, binaries, hiddenimports = collect_all("PIL")
datas = [entry for entry in datas if _keep_entry(entry)]
binaries = [entry for entry in binaries if _keep_entry(entry)]
hiddenimports = [name for name in hiddenimports if "avif" not in name.lower()]
