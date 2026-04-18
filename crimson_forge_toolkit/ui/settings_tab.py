from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from crimson_forge_toolkit.constants import (
    DEFAULT_UI_DATA_FONT_SIZE,
    DEFAULT_UI_DENSITY,
    DEFAULT_UI_FONT_SIZE,
    DEFAULT_UI_FONT_FAMILY,
    DEFAULT_UI_LOG_FONT_BOLD,
    DEFAULT_UI_LOG_FONT_FAMILY,
    DEFAULT_UI_LOG_FONT_SIZE,
    DEFAULT_UI_THEME,
    LOG_FONT_FAMILY_OPTIONS,
    UI_FONT_SIZE_MAX,
    UI_FONT_SIZE_MIN,
    UI_FONT_FAMILY_OPTIONS,
)
from crimson_forge_toolkit.ui.themes import UI_THEME_SCHEMES


class SettingsTab(QWidget):
    theme_changed = Signal(str)
    crash_capture_changed = Signal(bool)

    def __init__(self, *, settings, theme_key: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self._settings_ready = False
        self._settings_save_timer = QTimer(self)
        self._settings_save_timer.setSingleShot(True)
        self._settings_save_timer.setInterval(250)
        self._settings_save_timer.timeout.connect(self._save_settings)
        self._appearance_apply_timer = QTimer(self)
        self._appearance_apply_timer.setSingleShot(True)
        self._appearance_apply_timer.setInterval(140)
        self._appearance_apply_timer.timeout.connect(self._apply_pending_appearance_change)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        summary = QLabel(
            "Persistent global preferences for startup behavior, archive loading, UI layout memory, and safety prompts."
        )
        summary.setWordWrap(True)
        summary.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
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
        self.ui_font_family_combo = QComboBox()
        for family in UI_FONT_FAMILY_OPTIONS:
            self.ui_font_family_combo.addItem(family, family)
        appearance_layout.addRow("Global font family", self.ui_font_family_combo)
        self.density_combo = QComboBox()
        self.density_combo.addItem("Compact", "compact")
        self.density_combo.addItem("Normal", "normal")
        self.density_combo.addItem("Comfortable", "comfortable")
        appearance_layout.addRow("Density", self.density_combo)
        self.ui_font_size_spin = QSpinBox()
        self.ui_font_size_spin.setRange(UI_FONT_SIZE_MIN, UI_FONT_SIZE_MAX)
        self.ui_font_size_spin.setSuffix(" px")
        self.ui_font_size_spin.setKeyboardTracking(False)
        self.ui_font_size_spin.setAccelerated(True)
        self.ui_font_size_spin.setToolTip(
            f"Global UI font size. Minimum {UI_FONT_SIZE_MIN} px, maximum {UI_FONT_SIZE_MAX} px."
        )
        appearance_layout.addRow(
            f"Global font size ({UI_FONT_SIZE_MIN}-{UI_FONT_SIZE_MAX} px)",
            self.ui_font_size_spin,
        )
        self.data_font_size_spin = QSpinBox()
        self.data_font_size_spin.setRange(UI_FONT_SIZE_MIN, UI_FONT_SIZE_MAX)
        self.data_font_size_spin.setSuffix(" px")
        self.data_font_size_spin.setKeyboardTracking(False)
        self.data_font_size_spin.setAccelerated(True)
        self.data_font_size_spin.setToolTip(
            f"Used for dense lists, trees, tables, and column-heavy views. Minimum {UI_FONT_SIZE_MIN} px, maximum {UI_FONT_SIZE_MAX} px."
        )
        appearance_layout.addRow(
            f"Lists / columns font size ({UI_FONT_SIZE_MIN}-{UI_FONT_SIZE_MAX} px)",
            self.data_font_size_spin,
        )
        self.log_font_family_combo = QComboBox()
        for family in LOG_FONT_FAMILY_OPTIONS:
            self.log_font_family_combo.addItem(family, family)
        self.log_font_family_combo.setToolTip("Used for logs and code/text preview panes.")
        appearance_layout.addRow("Log / code font", self.log_font_family_combo)
        self.log_font_size_spin = QSpinBox()
        self.log_font_size_spin.setRange(8, 18)
        self.log_font_size_spin.setSuffix(" px")
        self.log_font_size_spin.setKeyboardTracking(False)
        self.log_font_size_spin.setAccelerated(True)
        appearance_layout.addRow("Log / code size", self.log_font_size_spin)
        self.log_font_bold_checkbox = QCheckBox("Bold emphasis in logs / code")
        self.log_font_bold_checkbox.setToolTip(
            "Controls whether highlighted log/code tokens use bold emphasis."
        )
        appearance_layout.addRow("", self.log_font_bold_checkbox)
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
        self.capture_crash_details_checkbox = QCheckBox(
            "Capture crash details to local report files on unhandled exceptions"
        )
        safety_layout.addWidget(self.confirm_workflow_cleanup_checkbox)
        safety_layout.addWidget(self.confirm_archive_cleanup_checkbox)
        safety_layout.addWidget(self.capture_crash_details_checkbox)
        root_layout.addWidget(safety_group)

        notes = QLabel(
            "These preferences are stored in the local config beside the EXE and apply across sessions."
        )
        notes.setWordWrap(True)
        notes.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        notes.setObjectName("HintLabel")
        root_layout.addWidget(notes)
        root_layout.addStretch(1)

        self.theme_combo.currentIndexChanged.connect(self._handle_appearance_changed)
        self.ui_font_family_combo.currentIndexChanged.connect(self._handle_appearance_changed)
        self.density_combo.currentIndexChanged.connect(self._handle_appearance_changed)
        self.ui_font_size_spin.valueChanged.connect(self._handle_appearance_changed)
        self.data_font_size_spin.valueChanged.connect(self._handle_appearance_changed)
        self.log_font_family_combo.currentIndexChanged.connect(self._handle_appearance_changed)
        self.log_font_size_spin.valueChanged.connect(self._handle_appearance_changed)
        self.log_font_bold_checkbox.toggled.connect(self._handle_appearance_changed)
        for checkbox in (
            self.auto_load_archive_checkbox,
            self.prefer_cache_checkbox,
            self.restore_last_tab_checkbox,
            self.remember_splitters_checkbox,
            self.confirm_workflow_cleanup_checkbox,
            self.confirm_archive_cleanup_checkbox,
            self.capture_crash_details_checkbox,
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
        self.sync_appearance_controls(theme_key)
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
        self.capture_crash_details_checkbox.setChecked(
            self._read_bool("preferences/capture_crash_details", False)
        )
        self._apply_checkbox_states()

    def _save_settings(self) -> None:
        if not self._settings_ready:
            return
        self.settings.setValue("appearance/theme", self.current_theme_key())
        self.settings.setValue("appearance/ui_font_family", self.current_ui_font_family())
        self.settings.setValue("appearance/ui_density", self.current_density_key())
        self.settings.setValue("appearance/ui_font_size", self.current_ui_font_size())
        self.settings.setValue("appearance/data_font_size", self.current_data_font_size())
        self.settings.setValue("appearance/log_font_family", self.current_log_font_family())
        self.settings.setValue("appearance/log_font_size", self.current_log_font_size())
        self.settings.setValue("appearance/log_font_bold", self.current_log_font_bold())
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
        previous_capture_value = self._read_bool("preferences/capture_crash_details", False)
        current_capture_value = self.capture_crash_details_checkbox.isChecked()
        self.settings.setValue("preferences/capture_crash_details", current_capture_value)
        self.settings.sync()
        self._apply_checkbox_states()
        if previous_capture_value != current_capture_value:
            self.crash_capture_changed.emit(current_capture_value)

    def schedule_settings_save(self, *_args) -> None:
        if not self._settings_ready:
            return
        self._settings_save_timer.start()

    def flush_settings_save(self) -> None:
        if self._appearance_apply_timer.isActive():
            self._appearance_apply_timer.stop()
            self._apply_pending_appearance_change()
            return
        if self._settings_save_timer.isActive():
            self._settings_save_timer.stop()
        self._save_settings()

    def _apply_checkbox_states(self) -> None:
        self.prefer_cache_checkbox.setEnabled(self.auto_load_archive_checkbox.isChecked())

    def _handle_appearance_changed(self) -> None:
        if not self._settings_ready:
            return
        self._appearance_apply_timer.start()

    def _apply_pending_appearance_change(self) -> None:
        if not self._settings_ready:
            return
        self._save_settings()
        self.theme_changed.emit(self.current_theme_key())

    def set_theme_selection(self, theme_key: str) -> None:
        resolved_theme_key = theme_key if theme_key in UI_THEME_SCHEMES else DEFAULT_UI_THEME
        index = self.theme_combo.findData(resolved_theme_key)
        if index < 0:
            index = self.theme_combo.findData(DEFAULT_UI_THEME)
        self.theme_combo.blockSignals(True)
        self.theme_combo.setCurrentIndex(max(0, index))
        self.theme_combo.blockSignals(False)

    def sync_appearance_controls(self, theme_key: str) -> None:
        resolved_theme_key = theme_key if theme_key in UI_THEME_SCHEMES else DEFAULT_UI_THEME
        ui_font_family = str(self.settings.value("appearance/ui_font_family", DEFAULT_UI_FONT_FAMILY) or DEFAULT_UI_FONT_FAMILY)
        density_key = str(self.settings.value("appearance/ui_density", DEFAULT_UI_DENSITY) or DEFAULT_UI_DENSITY)
        try:
            ui_font_size = int(self.settings.value("appearance/ui_font_size", DEFAULT_UI_FONT_SIZE))
        except (TypeError, ValueError):
            ui_font_size = DEFAULT_UI_FONT_SIZE
        try:
            data_font_size = int(self.settings.value("appearance/data_font_size", DEFAULT_UI_DATA_FONT_SIZE))
        except (TypeError, ValueError):
            data_font_size = DEFAULT_UI_DATA_FONT_SIZE
        log_font_family = str(
            self.settings.value("appearance/log_font_family", DEFAULT_UI_LOG_FONT_FAMILY)
            or DEFAULT_UI_LOG_FONT_FAMILY
        )
        try:
            log_font_size = int(self.settings.value("appearance/log_font_size", DEFAULT_UI_LOG_FONT_SIZE))
        except (TypeError, ValueError):
            log_font_size = DEFAULT_UI_LOG_FONT_SIZE
        log_font_bold = self._read_bool("appearance/log_font_bold", DEFAULT_UI_LOG_FONT_BOLD)
        self.set_theme_selection(resolved_theme_key)
        family_index = self.ui_font_family_combo.findData(ui_font_family)
        if family_index < 0:
            family_index = self.ui_font_family_combo.findData(DEFAULT_UI_FONT_FAMILY)
        self.ui_font_family_combo.blockSignals(True)
        self.ui_font_family_combo.setCurrentIndex(max(0, family_index))
        self.ui_font_family_combo.blockSignals(False)
        density_index = self.density_combo.findData(density_key)
        if density_index < 0:
            density_index = self.density_combo.findData(DEFAULT_UI_DENSITY)
        self.density_combo.blockSignals(True)
        self.density_combo.setCurrentIndex(max(0, density_index))
        self.density_combo.blockSignals(False)
        self.ui_font_size_spin.blockSignals(True)
        self.ui_font_size_spin.setValue(max(UI_FONT_SIZE_MIN, min(UI_FONT_SIZE_MAX, ui_font_size)))
        self.ui_font_size_spin.blockSignals(False)
        self.data_font_size_spin.blockSignals(True)
        self.data_font_size_spin.setValue(max(UI_FONT_SIZE_MIN, min(UI_FONT_SIZE_MAX, data_font_size)))
        self.data_font_size_spin.blockSignals(False)
        log_family_index = self.log_font_family_combo.findData(log_font_family)
        if log_family_index < 0:
            log_family_index = self.log_font_family_combo.findData(DEFAULT_UI_LOG_FONT_FAMILY)
        self.log_font_family_combo.blockSignals(True)
        self.log_font_family_combo.setCurrentIndex(max(0, log_family_index))
        self.log_font_family_combo.blockSignals(False)
        self.log_font_size_spin.blockSignals(True)
        self.log_font_size_spin.setValue(max(8, min(18, log_font_size)))
        self.log_font_size_spin.blockSignals(False)
        self.log_font_bold_checkbox.blockSignals(True)
        self.log_font_bold_checkbox.setChecked(bool(log_font_bold))
        self.log_font_bold_checkbox.blockSignals(False)

    def current_theme_key(self) -> str:
        data = self.theme_combo.currentData()
        return str(data) if data is not None else DEFAULT_UI_THEME

    def current_density_key(self) -> str:
        data = self.density_combo.currentData()
        return str(data) if data is not None else DEFAULT_UI_DENSITY

    def current_ui_font_family(self) -> str:
        data = self.ui_font_family_combo.currentData()
        return str(data) if data is not None else DEFAULT_UI_FONT_FAMILY

    def current_ui_font_size(self) -> int:
        return int(self.ui_font_size_spin.value())

    def current_data_font_size(self) -> int:
        return int(self.data_font_size_spin.value())

    def current_log_font_family(self) -> str:
        data = self.log_font_family_combo.currentData()
        return str(data) if data is not None else DEFAULT_UI_LOG_FONT_FAMILY

    def current_log_font_size(self) -> int:
        return int(self.log_font_size_spin.value())

    def current_log_font_bold(self) -> bool:
        return bool(self.log_font_bold_checkbox.isChecked())
