"""
ui_main.py
Main window: one-click data collection, live chart, and end-of-run
soaking / alignment recipe recommendation (+ local AI commentary).
"""

import csv
import sys
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QSettings, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QGroupBox, QFormLayout, QTextEdit, QFileDialog,
    QMessageBox, QHeaderView, QLineEdit,
)

from analysis import SoakAnalysisEngine
from ai_recipe import (
    AISettings, PROVIDERS, DEFAULT_MODELS, PROVIDER_CLAUDE, PROVIDER_COPILOT,
    MANUAL_PROVIDERS, request_ai_recipe, cross_check,
    build_manual_prompt, parse_manual_reply,
)
from data_sources import SOURCE_REGISTRY, FileDataSource, ManualEntryDataSource


DEG_C = "\u00b0C"      # constant unit symbol shown next to the temperature


class AIWorker(QThread):
    """Asks the AI for a recipe off the GUI thread so the window never freezes."""
    done = Signal(int, object, str)   # run_id, AIRecipe | None, error text

    def __init__(self, run_id, t, y, fit, threshold_um, max_slope, settings,
                 temperature_c=None):
        super().__init__()
        self.run_id = run_id
        self._args = (t, y, fit, threshold_um, max_slope, settings, temperature_c)

    def run(self):
        recipe, err = request_ai_recipe(*self._args)
        self.done.emit(self.run_id, recipe, err)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Needle Soak & Alignment Recipe Generator — Precio XL")
        self.resize(1280, 760)

        self.engine = SoakAnalysisEngine(saturation_threshold_um=10.0)
        self.data_source = None
        self._last_recommendation = None
        self._last_ai = None            # AIRecipe
        self._last_ai_check = ""
        self._last_ai_error = ""
        self._ai_workers = []           # keep running threads alive until they finish
        self._manual_prompt = ""
        self._manual_last_t = 0.0
        self._ai_run_id = 0

        self._build_ui()
        self._select_source(self.source_combo.currentText())

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # ---- Left: controls ----
        left = QVBoxLayout()
        root.addLayout(left, 0)

        meta_box = QGroupBox("Run Info")
        meta_form = QFormLayout(meta_box)
        self.probe_card_edit = QComboBox()
        self.probe_card_edit.setEditable(True)
        self.probe_card_edit.addItems(["TP-Vertical-001", "TP-Vertical-002"])
        self.lot_edit = QComboBox()
        self.lot_edit.setEditable(True)
        meta_form.addRow("Probe card:", self.probe_card_edit)
        meta_form.addRow("Lot / wafer ID:", self.lot_edit)
        self.temp_combo = QComboBox()
        for deg in range(-40, 126, 5):              # -40 ... 125, 5-degree steps
            self.temp_combo.addItem(str(deg), deg)
        self.temp_combo.setCurrentIndex(self.temp_combo.findData(25))   # default 25
        temp_row = QWidget()
        temp_layout = QHBoxLayout(temp_row)
        temp_layout.setContentsMargins(0, 0, 0, 0)
        temp_layout.addWidget(self.temp_combo, 1)
        temp_layout.addWidget(QLabel(DEG_C))        # constant unit symbol
        meta_form.addRow("Temperature:", temp_row)
        left.addWidget(meta_box)

        source_box = QGroupBox("Data Source")
        source_layout = QVBoxLayout(source_box)
        self.source_combo = QComboBox()
        self.source_combo.addItems(list(SOURCE_REGISTRY.keys()))
        self.source_combo.currentTextChanged.connect(self._select_source)
        source_layout.addWidget(self.source_combo)
        left.addWidget(source_box)

        threshold_box = QGroupBox("Recipe Parameters")
        threshold_form = QFormLayout(threshold_box)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(1.0, 100.0)
        self.threshold_spin.setValue(10.0)
        self.threshold_spin.setSuffix(" um")
        self.threshold_spin.valueChanged.connect(self._on_threshold_changed)
        threshold_form.addRow("Saturation threshold:", self.threshold_spin)
        left.addWidget(threshold_box)

        ai_box = QGroupBox("AI Recipe Settings")
        ai_form = QFormLayout(ai_box)
        self.ai_provider_combo = QComboBox()
        self.ai_provider_combo.addItems(PROVIDERS)
        self.ai_model_edit = QLineEdit()
        self.ai_key_edit = QLineEdit()
        self.ai_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.ai_key_edit.setPlaceholderText("sk-ant-...  (not saved to disk)")
        ai_form.addRow("Provider:", self.ai_provider_combo)
        ai_form.addRow("Model:", self.ai_model_edit)
        ai_form.addRow("API key:", self.ai_key_edit)
        self.ai_note = QLabel("Copilot: copy/paste, no key needed. "
                              "Claude: readings go to the cloud. "
                              "Ollama: stays on this PC.")
        self.ai_note.setWordWrap(True)
        ai_form.addRow(self.ai_note)
        left.addWidget(ai_box)
        self.ai_provider_combo.currentTextChanged.connect(self._on_provider_changed)

        control_box = QGroupBox("Control")
        control_layout = QVBoxLayout(control_box)
        self.start_btn = QPushButton("▶  Start Data Collection")
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setEnabled(False)
        self.end_run_btn = QPushButton("✓  End Run && Generate Recipe")
        self.end_run_btn.clicked.connect(self._on_end_run)
        self.reset_btn = QPushButton("↺  Reset")
        self.reset_btn.clicked.connect(self._on_reset)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.end_run_btn)
        control_layout.addWidget(self.reset_btn)
        left.addWidget(control_box)

        manual_box = QGroupBox("Manual Reading Entry")
        manual_layout = QHBoxLayout(manual_box)
        self.manual_spin = QDoubleSpinBox()
        self.manual_spin.setRange(0.0, 1000.0)
        self.manual_spin.setSuffix(" um")
        self.manual_add_btn = QPushButton("Add Reading")
        self.manual_add_btn.clicked.connect(self._on_manual_add)
        manual_layout.addWidget(self.manual_spin)
        manual_layout.addWidget(self.manual_add_btn)
        left.addWidget(manual_box)

        self.status_label = QLabel("Idle.")
        self.status_label.setWordWrap(True)
        self.status_label.setMaximumWidth(260)
        left.addWidget(self.status_label)

        left.addStretch(1)

        # ---- Middle: chart + table ----
        middle = QVBoxLayout()
        root.addLayout(middle, 2)

        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        self.plot = pg.PlotWidget(title="Needle Height Diff vs Elapsed Soak Time")
        self.plot.setLabel("bottom", "Elapsed time", units="min")
        self.plot.setLabel("left", "Height diff", units="um")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.scatter = self.plot.plot([], [], pen=None, symbol="o",
                                      symbolBrush="b", symbolSize=8, name="Readings")
        self.fit_curve = self.plot.plot([], [], pen=pg.mkPen("r", width=2), name="Fitted curve")
        self.threshold_line = pg.InfiniteLine(
            pos=10.0, angle=0, pen=pg.mkPen("g", style=Qt.PenStyle.DashLine))
        self.plot.addItem(self.threshold_line)
        middle.addWidget(self.plot, 3)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Elapsed (min)", "Height diff (um)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        middle.addWidget(self.table, 2)

        # ---- Right: recipe output ----
        right = QVBoxLayout()
        root.addLayout(right, 1)

        right.addWidget(QLabel("<b>AI Recipe Suggestion</b>"))
        self.aiTextBox = QTextEdit()
        self.aiTextBox.setReadOnly(True)
        self.aiTextBox.setStyleSheet("font-family: Consolas, 'Courier New', monospace;")
        right.addWidget(self.aiTextBox, 3)

        self.manual_box = QGroupBox("Microsoft Copilot - copy / paste")
        manual_v = QVBoxLayout(self.manual_box)
        steps = QLabel("1) Click <b>Copy prompt</b>  2) Paste it into Copilot and send  "
                       "3) Copy Copilot's reply and paste it below  4) Click <b>Apply reply</b>")
        steps.setWordWrap(True)
        manual_v.addWidget(steps)
        self.copy_prompt_btn = QPushButton("Copy prompt for Copilot")
        self.copy_prompt_btn.setEnabled(False)
        self.copy_prompt_btn.clicked.connect(self._on_copy_prompt)
        manual_v.addWidget(self.copy_prompt_btn)
        self.reply_edit = QTextEdit()
        self.reply_edit.setPlaceholderText("Paste Copilot's reply here ...")
        self.reply_edit.setFixedHeight(90)
        manual_v.addWidget(self.reply_edit)
        self.apply_reply_btn = QPushButton("Apply reply")
        self.apply_reply_btn.setEnabled(False)
        self.apply_reply_btn.clicked.connect(self._on_apply_reply)
        manual_v.addWidget(self.apply_reply_btn)
        right.addWidget(self.manual_box)

        self.export_btn = QPushButton("Export Report (CSV)")
        self.export_btn.clicked.connect(self._on_export)
        self.export_btn.setEnabled(False)
        right.addWidget(self.export_btn)

        self._load_ai_settings()   # needs both the AI settings and the manual box

    def _temperature_c(self):
        return self.temp_combo.currentData()

    # ------------------------------------------------------------ AI settings
    def _qsettings(self):
        return QSettings("NeedleSoak", "RecipeGenerator")

    def _load_ai_settings(self):
        q = self._qsettings()
        provider = q.value("ai/provider", PROVIDER_COPILOT)
        if provider not in PROVIDERS:
            provider = PROVIDER_COPILOT
        self.ai_provider_combo.blockSignals(True)
        self.ai_provider_combo.setCurrentText(provider)
        self.ai_provider_combo.blockSignals(False)
        self.ai_model_edit.setText(q.value(f"ai/model/{provider}", ""))
        self._apply_provider_widgets(provider)

    def _apply_provider_widgets(self, provider: str):
        manual = provider in MANUAL_PROVIDERS
        self.ai_model_edit.setPlaceholderText(DEFAULT_MODELS.get(provider, ""))
        self.ai_model_edit.setEnabled(not manual)
        self.ai_key_edit.setEnabled(provider == PROVIDER_CLAUDE)
        self.manual_box.setVisible(manual)

    def _on_provider_changed(self, provider: str):
        self.ai_model_edit.setText(self._qsettings().value(f"ai/model/{provider}", ""))
        self._apply_provider_widgets(provider)

    def _save_ai_settings(self):
        q = self._qsettings()
        provider = self.ai_provider_combo.currentText()
        q.setValue("ai/provider", provider)
        q.setValue(f"ai/model/{provider}", self.ai_model_edit.text().strip())
        # The API key is deliberately NOT stored.

    def _ai_settings(self) -> AISettings:
        return AISettings(provider=self.ai_provider_combo.currentText(),
                          model=self.ai_model_edit.text(),
                          api_key=self.ai_key_edit.text())

    # ------------------------------------------------------------- Source
    def _select_source(self, name: str):
        if self.data_source is not None:
            self.data_source.stop()
            self.data_source.reading.disconnect()
            self.data_source.status.disconnect()
            self.data_source.error.disconnect()
            self.data_source.finished.disconnect()

        self.data_source = SOURCE_REGISTRY[name]()
        self.data_source.reading.connect(self._on_new_reading)
        self.data_source.status.connect(self._on_status)
        self.data_source.error.connect(self._on_error)
        self.data_source.finished.connect(self._set_idle_buttons)

        is_manual = isinstance(self.data_source, ManualEntryDataSource)
        self.manual_spin.setEnabled(is_manual)
        self.manual_add_btn.setEnabled(is_manual)
        self._set_idle_buttons()

        if is_manual:
            self.status_label.setText("Manual entry: type a reading and press Add Reading.")
        elif isinstance(self.data_source, FileDataSource):
            self.status_label.setText("File upload: press Start and choose an Excel / CSV / TXT file.")
        else:
            self.status_label.setText(f"Source: {name}. Press Start to begin.")

    def _on_threshold_changed(self, val):
        self.engine.saturation_threshold_um = val
        self.threshold_line.setPos(val)

    def _set_idle_buttons(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    # ------------------------------------------------------------ Control
    def _clear_results(self):
        self._ai_run_id += 1          # ignore any AI answer still on its way
        self.engine.clear()
        self.table.setRowCount(0)
        self.scatter.setData([], [])
        self.fit_curve.setData([], [])
        self.aiTextBox.clear()
        self.export_btn.setEnabled(False)
        self._last_recommendation = None
        self._last_ai = None
        self._last_ai_check = ""
        self._last_ai_error = ""
        self._manual_prompt = ""
        self.reply_edit.clear()
        self.copy_prompt_btn.setEnabled(False)
        self.apply_reply_btn.setEnabled(False)

    def _on_start(self):
        # File source: ask which file to load
        if isinstance(self.data_source, FileDataSource):
            path, _ = QFileDialog.getOpenFileName(
                self, "Select profile data file", "",
                "Data files (*.xlsx *.xls *.csv *.txt);;All files (*)")
            if not path:
                return
            self.data_source.file_path = path

        self._clear_results()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.data_source.start()

    def _on_stop(self):
        if self.data_source is not None and self.data_source.is_running:
            self.data_source.stop()
        self._set_idle_buttons()

    def _on_reset(self):
        self._on_stop()
        self._clear_results()
        self.status_label.setText("Reset.")

    def _on_manual_add(self):
        if not isinstance(self.data_source, ManualEntryDataSource):
            QMessageBox.information(self, "Manual entry",
                                    "Switch the Data Source to 'Manual entry' to use this field.")
            return
        if not self.data_source.is_running:
            self._clear_results()
            self.data_source.start()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
        self.data_source.submit_reading(self.manual_spin.value())

    # ------------------------------------------------------------- Events
    def _on_new_reading(self, elapsed_min: float, height_diff_um: float):
        self.engine.add_reading(elapsed_min, height_diff_um)

        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(f"{elapsed_min:.1f}"))
        self.table.setItem(row, 1, QTableWidgetItem(f"{height_diff_um:.2f}"))

        t, y = self.engine.as_arrays()
        self.scatter.setData(t, y)

        fit = self.engine.fit()
        if fit.success:
            t_grid = np.linspace(0, max(t[-1] * 1.3, 10), 200)
            self.fit_curve.setData(t_grid, fit.predict(t_grid))

        self.status_label.setText(
            f"Last reading: t={elapsed_min:.1f} min, diff={height_diff_um:.2f} um "
            f"({len(self.engine.readings)} points collected)"
        )

    def _on_status(self, msg: str):
        self.status_label.setText(msg)

    def _on_error(self, msg: str):
        QMessageBox.warning(self, "Data source error", msg)
        self._set_idle_buttons()

    # ------------------------------------------------------------ Recipe
    def _on_end_run(self):
        if len(self.engine.readings) < 2:
            QMessageBox.information(self, "Not enough data",
                                    "Collect at least a few readings before generating a recipe.")
            return
        self._on_stop()

        rec = self.engine.recommend()
        self._last_recommendation = rec
        fit = rec.fit

        self.export_btn.setEnabled(True)

        # ---- AI recipe suggestion (background thread)
        if len(self.engine.readings) < 3:
            self.aiTextBox.setPlainText("AI recipe needs at least 3 readings.")
            return

        self._save_ai_settings()
        settings = self._ai_settings()
        t, y = self.engine.as_arrays()

        if settings.provider in MANUAL_PROVIDERS:
            self._manual_prompt = build_manual_prompt(
                t, y, fit, self.engine.saturation_threshold_um,
                self.engine.max_stable_slope_um_per_min, self._temperature_c())
            self._manual_last_t = float(t[-1])
            self.reply_edit.clear()
            self.copy_prompt_btn.setEnabled(True)
            self.apply_reply_btn.setEnabled(True)
            self.aiTextBox.setPlainText(
                "Copilot mode: click 'Copy prompt for Copilot' below, paste it into "
                "Copilot, then paste Copilot's reply into the box and click 'Apply reply'.")
            return

        self._ai_run_id += 1
        self.aiTextBox.setPlainText(
            f"Asking the AI ({settings.provider}, {settings.resolved_model()}) "
            "for a recipe ... please wait.")
        worker = AIWorker(self._ai_run_id, t, y, fit,
                          self.engine.saturation_threshold_um,
                          self.engine.max_stable_slope_um_per_min, settings,
                          self._temperature_c())
        worker.done.connect(self._on_ai_done)
        # Drop the reference only once the thread has really finished, otherwise
        # Qt aborts ("QThread: Destroyed while thread is still running").
        worker.finished.connect(lambda w=worker: self._release_worker(w))
        self._ai_workers.append(worker)
        worker.start()

    def _release_worker(self, worker):
        if worker in self._ai_workers:
            self._ai_workers.remove(worker)
        worker.deleteLater()

    def _on_copy_prompt(self):
        if not self._manual_prompt:
            return
        QApplication.clipboard().setText(self._manual_prompt)
        self.status_label.setText("Prompt copied. Paste it into Copilot (Ctrl+V).")

    def _on_apply_reply(self):
        recipe, err = parse_manual_reply(self.reply_edit.toPlainText(),
                                         self._manual_last_t)
        if recipe is None:
            self._last_ai_error = err
            self.aiTextBox.setPlainText("AI recipe not applied:\n" + err)
            return
        self._last_ai_error = ""
        self._show_ai_recipe(recipe)

    def _ai_header_lines(self) -> list:
        temp = self._temperature_c()
        return [
            f"Probe card : {self.probe_card_edit.currentText() or '(not set)'}",
            f"Lot / Wafer: {self.lot_edit.currentText() or '(not set)'}",
            f"Temperature: {temp} {DEG_C}" if temp is not None else "Temperature: (not set)",
            f"Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ]

    def _show_ai_recipe(self, recipe):
        self._last_ai = recipe
        self._last_ai_check = cross_check(recipe, self._last_recommendation)
        self.aiTextBox.setPlainText(
            recipe.as_text(self.engine.saturation_threshold_um, self._ai_header_lines())
            + "\n\nCROSS-CHECK: " + self._last_ai_check)

    def _on_ai_done(self, run_id: int, recipe, error: str):
        if run_id != self._ai_run_id:
            return   # a newer run / reset happened; discard
        if recipe is None:
            self._last_ai_error = error
            self.aiTextBox.setPlainText("AI recipe unavailable:\n" + error)
            return
        self._show_ai_recipe(recipe)

    # ------------------------------------------------------------- Export
    def _on_export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export report",
                                              "soak_recipe_report.csv", "CSV files (*.csv)")
        if not path:
            return
        rec = self._last_recommendation
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["Probe card", self.probe_card_edit.currentText()])
            writer.writerow(["Lot/wafer", self.lot_edit.currentText()])
            writer.writerow(["Temperature (degC)", self._temperature_c()])
            writer.writerow(["Generated", datetime.now().isoformat(timespec="seconds")])
            writer.writerow([])
            writer.writerow(["Elapsed (min)", "Height diff (um)"])
            for r in self.engine.readings:
                writer.writerow([r.elapsed_min, r.height_diff_um])
            writer.writerow([])
            if rec and rec.fit:
                writer.writerow(["Fit A (um)", round(rec.fit.A, 3)])
                writer.writerow(["Fit tau (min)", round(rec.fit.tau, 3)])
                writer.writerow(["Fit C (um)", round(rec.fit.C, 3)])
                writer.writerow(["Fit R2", round(rec.fit.r2, 4)])
            ai = self._last_ai
            if ai:
                writer.writerow([])
                writer.writerow(["AI provider / model", f"{ai.provider} / {ai.model}"])
                writer.writerow(["AI soak time (min)", ai.soak_time_min])
                writer.writerow(["AI saturated", ai.saturated])
                writer.writerow(["AI alignment checks (min)",
                                 ";".join(str(x) for x in ai.alignment_checks_min)])
                writer.writerow(["AI confidence", ai.confidence])
                writer.writerow(["AI interpretation", ai.interpretation])
                writer.writerow(["AI production recommendation", ai.production_recommendation])
                writer.writerow(["AI risk if reduced", ai.risk_if_reduced])
                writer.writerow(["Cross-check vs curve fit", self._last_ai_check])
            elif self._last_ai_error:
                writer.writerow(["AI recipe", "unavailable: " + self._last_ai_error])
        QMessageBox.information(self, "Exported", f"Report saved to:\n{path}")

    # -------------------------------------------------------------- Close
    def closeEvent(self, event):
        if self.data_source is not None and self.data_source.is_running:
            self.data_source.stop()
        for worker in list(self._ai_workers):
            if worker.isRunning():
                worker.wait(3000)
                if worker.isRunning():
                    worker.terminate()
                    worker.wait()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
