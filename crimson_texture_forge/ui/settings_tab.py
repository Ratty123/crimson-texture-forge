from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from crimson_texture_forge.constants import DEFAULT_UI_THEME
from crimson_texture_forge.ui.themes import UI_THEME_SCHEMES


class SettingsTab(QWidget):
    theme_changed = Signal(str)

    def __init__(self, *, settings, theme_key: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self._settings_ready = False
        self._settings_save_timer = QTimer(self)
        self._settings_save_timer.setSingleShot(True)
        self._settings_save_timer.setInterval(250)
        self._settings_save_timer.timeout.connect(self._save_settings)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        summary = QLabel(
            "Persistent global preferences for startup behavior, archive loading, UI layout memory, and safety prompts."
        )
        summary.setWordWrap(True)
        summary.setObjectName("HintLabel")
        root_layout.addWidget(summary)

        appearance_group = QGroupBox("Appearance")
        appearance_layout = QFormLayout(appearance_group)
        appearance_layout.setContentsMargins(12, 14, 12, 12)
        appearance_layout.setHorizontalSpacing(12)
        appearance_layout.setVerticalSpacing(10)
        self.theme_combo = QComboBox()
        for key, theme in UI_THEME_SCHEMES.items():
            self.theme_combo.addItem(theme["label"], key)
        appearance_layout.addRow("Theme", self.theme_combo)
        root_layout.addWidget(appearance_group)

        startup_group = QGroupBox("Startup")
        startup_layout = QVBoxLayout(startup_group)
        startup_layout.setContentsMargins(12, 14, 12, 12)
        startup_layout.setSpacing(8)
        self.auto_load_archive_checkbox = QCheckBox("Auto-load Archive Browser on startup")
        self.prefer_cache_checkbox = QCheckBox("Prefer archive cache on startup")
        self.restore_last_tab_checkbox = QCheckBox("Restore last active tab")
        self.prefer_cache_checkbox.setToolTip(
            "When enabled, startup archive loading uses the saved cache when possible. Disable it to force a refresh."
        )
        startup_layout.addWidget(self.auto_load_archive_checkbox)
        startup_layout.addWidget(self.prefer_cache_checkbox)
        startup_layout.addWidget(self.restore_last_tab_checkbox)
        root_layout.addWidget(startup_group)

        layout_group = QGroupBox("Layout")
        layout_layout = QVBoxLayout(layout_group)
        layout_layout.setContentsMargins(12, 14, 12, 12)
        layout_layout.setSpacing(8)
        self.remember_splitters_checkbox = QCheckBox("Remember pane sizes and splitters")
        layout_layout.addWidget(self.remember_splitters_checkbox)
        root_layout.addWidget(layout_group)

        safety_group = QGroupBox("Safety")
        safety_layout = QVBoxLayout(safety_group)
        safety_layout.setContentsMargins(12, 14, 12, 12)
        safety_layout.setSpacing(8)
        self.confirm_workflow_cleanup_checkbox = QCheckBox("Confirm clearing PNG / DDS output folders before Start")
        self.confirm_archive_cleanup_checkbox = QCheckBox("Confirm clearing archive extraction target")
        safety_layout.addWidget(self.confirm_workflow_cleanup_checkbox)
        safety_layout.addWidget(self.confirm_archive_cleanup_checkbox)
        root_layout.addWidget(safety_group)

        notes = QLabel(
            "These preferences are stored in the local config beside the EXE and apply across sessions."
        )
        notes.setWordWrap(True)
        notes.setObjectName("HintLabel")
        root_layout.addWidget(notes)
        root_layout.addStretch(1)

        self.theme_combo.currentIndexChanged.connect(self._handle_theme_combo_changed)
        for checkbox in (
            self.auto_load_archive_checkbox,
            self.prefer_cache_checkbox,
            self.restore_last_tab_checkbox,
            self.remember_splitters_checkbox,
            self.confirm_workflow_cleanup_checkbox,
            self.confirm_archive_cleanup_checkbox,
        ):
            checkbox.toggled.connect(self.schedule_settings_save)

        self._load_settings(theme_key)
        self._settings_ready = True

    def _read_bool(self, key: str, default: bool) -> bool:
        value = self.settings.value(key, default)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _load_settings(self, theme_key: str) -> None:
        resolved_theme_key = theme_key if theme_key in UI_THEME_SCHEMES else DEFAULT_UI_THEME
        self.set_theme_selection(resolved_theme_key)
        self.auto_load_archive_checkbox.setChecked(
            self._read_bool("preferences/auto_load_archive_on_startup", False)
        )
        self.prefer_cache_checkbox.setChecked(
            self._read_bool("preferences/prefer_archive_cache_on_startup", True)
        )
        self.restore_last_tab_checkbox.setChecked(
            self._read_bool("preferences/restore_last_active_tab", True)
        )
        self.remember_splitters_checkbox.setChecked(
            self._read_bool("preferences/remember_splitter_sizes", True)
        )
        self.confirm_workflow_cleanup_checkbox.setChecked(
            self._read_bool("preferences/confirm_workflow_output_cleanup", True)
        )
        self.confirm_archive_cleanup_checkbox.setChecked(
            self._read_bool("preferences/confirm_archive_extract_cleanup", True)
        )
        self._apply_checkbox_states()

    def _save_settings(self) -> None:
        if not self._settings_ready:
            return
        self.settings.setValue("preferences/auto_load_archive_on_startup", self.auto_load_archive_checkbox.isChecked())
        self.settings.setValue(
            "preferences/prefer_archive_cache_on_startup",
            self.prefer_cache_checkbox.isChecked(),
        )
        self.settings.setValue(
            "preferences/restore_last_active_tab",
            self.restore_last_tab_checkbox.isChecked(),
        )
        self.settings.setValue(
            "preferences/remember_splitter_sizes",
            self.remember_splitters_checkbox.isChecked(),
        )
        self.settings.setValue(
            "preferences/confirm_workflow_output_cleanup",
            self.confirm_workflow_cleanup_checkbox.isChecked(),
        )
        self.settings.setValue(
            "preferences/confirm_archive_extract_cleanup",
            self.confirm_archive_cleanup_checkbox.isChecked(),
        )
        self.settings.sync()
        self._apply_checkbox_states()

    def schedule_settings_save(self, *_args) -> None:
        if not self._settings_ready:
            return
        self._settings_save_timer.start()

    def flush_settings_save(self) -> None:
        if self._settings_save_timer.isActive():
            self._settings_save_timer.stop()
        self._save_settings()

    def _apply_checkbox_states(self) -> None:
        self.prefer_cache_checkbox.setEnabled(self.auto_load_archive_checkbox.isChecked())

    def _handle_theme_combo_changed(self) -> None:
        self._save_settings()
        theme_key = self.current_theme_key()
        self.settings.setValue("appearance/theme", theme_key)
        self.settings.sync()
        self.theme_changed.emit(theme_key)

    def set_theme_selection(self, theme_key: str) -> None:
        resolved_theme_key = theme_key if theme_key in UI_THEME_SCHEMES else DEFAULT_UI_THEME
        index = self.theme_combo.findData(resolved_theme_key)
        if index < 0:
            index = self.theme_combo.findData(DEFAULT_UI_THEME)
        self.theme_combo.blockSignals(True)
        self.theme_combo.setCurrentIndex(max(0, index))
        self.theme_combo.blockSignals(False)

    def current_theme_key(self) -> str:
        data = self.theme_combo.currentData()
        return str(data) if data is not None else DEFAULT_UI_THEME
