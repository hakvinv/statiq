"""PyQRS Clone — statistisches Werkzeug für Wahrscheinlichkeiten, Quantile und Stichproben."""
from __future__ import annotations

import sys
import math
import traceback

import numpy as np
from scipy import optimize

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QListWidget, QListWidgetItem, QLabel, QLineEdit, QFormLayout, QGroupBox,
    QPushButton, QRadioButton, QButtonGroup, QPlainTextEdit, QSpinBox,
    QDoubleSpinBox, QStackedWidget, QComboBox, QMessageBox, QTabWidget,
    QSlider, QCheckBox,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from distributions import ALL_DISTRIBUTIONS, DISCRETE, CONTINUOUS, Distribution, ParamSpec


# ---------------------------------------------------------------------------
# Slider linked to a (Double)SpinBox — supports continuous + integer values
# ---------------------------------------------------------------------------

SLIDER_STEPS = 1000


class LinkedSlider(QWidget):
    """A QSlider that stays in sync with a Q(Double)SpinBox.

    Slider position 0..SLIDER_STEPS maps linearly to [lo, hi].
    For integer spinboxes, the value is rounded.
    Emits valueChanged whenever either control updates.
    """

    valueChanged = pyqtSignal()

    def __init__(self, spinbox: QSpinBox | QDoubleSpinBox, lo: float, hi: float):
        super().__init__()
        self.spinbox = spinbox
        self.integer = isinstance(spinbox, QSpinBox)
        self.lo = float(lo)
        self.hi = float(hi)
        if self.hi <= self.lo:
            self.hi = self.lo + 1.0

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, SLIDER_STEPS)
        self.slider.setMinimumWidth(140)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.spinbox, 0)
        layout.addWidget(self.slider, 1)

        self._syncing = False
        self.slider.valueChanged.connect(self._on_slider)
        self.spinbox.valueChanged.connect(self._on_spin)
        self._sync_slider_from_spin()

    def set_range(self, lo: float, hi: float):
        self.lo, self.hi = float(lo), float(hi)
        if self.hi <= self.lo:
            self.hi = self.lo + 1.0
        self._sync_slider_from_spin()

    def _value_to_pos(self, v: float) -> int:
        if self.hi == self.lo:
            return 0
        return int(round((v - self.lo) / (self.hi - self.lo) * SLIDER_STEPS))

    def _pos_to_value(self, pos: int) -> float:
        v = self.lo + pos / SLIDER_STEPS * (self.hi - self.lo)
        if self.integer:
            v = round(v)
        return v

    def _sync_slider_from_spin(self):
        pos = max(0, min(SLIDER_STEPS, self._value_to_pos(self.spinbox.value())))
        self._syncing = True
        self.slider.setValue(pos)
        self._syncing = False

    def _on_slider(self, pos: int):
        if self._syncing:
            return
        v = self._pos_to_value(pos)
        self._syncing = True
        if self.integer:
            self.spinbox.setValue(int(v))
        else:
            self.spinbox.setValue(float(v))
        self._syncing = False
        self.valueChanged.emit()

    def _on_spin(self, _v):
        if self._syncing:
            return
        self._sync_slider_from_spin()
        self.valueChanged.emit()


# ---------------------------------------------------------------------------
# Plot canvas with PDF/PMF and CDF stacked
# ---------------------------------------------------------------------------

class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(6, 5), tight_layout=True)
        super().__init__(self.fig)
        self.ax_pdf = self.fig.add_subplot(211)
        self.ax_cdf = self.fig.add_subplot(212)

    def plot(self, dist: Distribution, rv, highlights: dict | None = None):
        self.ax_pdf.clear()
        self.ax_cdf.clear()
        
        # Apply dark theme base
        self.fig.patch.set_facecolor('#252526')
        for ax in (self.ax_pdf, self.ax_cdf):
            ax.set_facecolor('#252526')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#555555')
            ax.spines['left'].set_color('#555555')
            ax.tick_params(colors='#888888')
            ax.xaxis.label.set_color('#E0E0E0')
            ax.yaxis.label.set_color('#E0E0E0')
            ax.title.set_color('#E0E0E0')

        try:
            lo, hi = dist.support_range(rv)
        except Exception:
            lo, hi = -5, 5
        if hi <= lo:
            hi = lo + 1

        if dist.discrete:
            k = np.arange(int(math.floor(lo)), int(math.ceil(hi)) + 1)
            pmf = rv.pmf(k)
            self.ax_pdf.bar(k, pmf, width=0.8, color="#319BFF", edgecolor="#0A84FF")
            self.ax_pdf.set_ylabel("PMF")
            cdf = rv.cdf(k)
            self.ax_cdf.step(k, cdf, where="post", color="#319BFF", linewidth=2)
            self.ax_cdf.set_ylabel("CDF")
        else:
            xs = np.linspace(lo, hi, 600)
            pdf = rv.pdf(xs)
            self.ax_pdf.plot(xs, pdf, color="#319BFF", linewidth=2)
            self.ax_pdf.fill_between(xs, pdf, alpha=0.15, color="#319BFF")
            self.ax_pdf.set_ylabel("PDF")
            cdf = rv.cdf(xs)
            self.ax_cdf.plot(xs, cdf, color="#319BFF", linewidth=2)
            self.ax_cdf.set_ylabel("CDF")

        self.ax_pdf.set_title(dist.name)
        self.ax_cdf.set_xlabel("x")
        self.ax_cdf.set_ylim(-0.02, 1.02)

        if highlights:
            self._draw_highlights(dist, rv, highlights, lo, hi)

        for ax in (self.ax_pdf, self.ax_cdf):
            ax.grid(True, alpha=0.1, color="#FFFFFF", linestyle="--")

        self.draw()

    def _draw_highlights(self, dist, rv, h, lo, hi):
        mode = h.get("mode")
        if mode in ("le", "lt"):
            x = h["x"]
            if dist.discrete:
                k = np.arange(int(math.floor(lo)), int(x) + 1)
                self.ax_pdf.bar(k, rv.pmf(k), width=0.8, color="#FF453A", edgecolor="#FF6961")
            else:
                xs = np.linspace(lo, x, 300)
                self.ax_pdf.fill_between(xs, rv.pdf(xs), alpha=0.4, color="#FF453A")
            self._mark_x(x, rv.cdf(x))
        elif mode == "ge":
            x = h["x"]
            if dist.discrete:
                k = np.arange(int(x), int(math.ceil(hi)) + 1)
                self.ax_pdf.bar(k, rv.pmf(k), width=0.8, color="#FF453A", edgecolor="#FF6961")
            else:
                xs = np.linspace(x, hi, 300)
                self.ax_pdf.fill_between(xs, rv.pdf(xs), alpha=0.4, color="#FF453A")
            self._mark_x(x, rv.cdf(x))
        elif mode == "between":
            a, b = h["a"], h["b"]
            if dist.discrete:
                k = np.arange(int(a), int(b) + 1)
                self.ax_pdf.bar(k, rv.pmf(k), width=0.8, color="#FF453A", edgecolor="#FF6961")
            else:
                xs = np.linspace(a, b, 300)
                self.ax_pdf.fill_between(xs, rv.pdf(xs), alpha=0.4, color="#FF453A")
            self._mark_x(a, rv.cdf(a))
            self._mark_x(b, rv.cdf(b))
        elif mode == "sample":
            samples = h["samples"]
            ymax = self.ax_pdf.get_ylim()[1]
            self.ax_pdf.scatter(samples, np.full_like(samples, ymax * 0.02),
                                color="#FF453A", marker="|", s=200, zorder=5)

    def _mark_x(self, x, cdf_val):
        for ax in (self.ax_pdf, self.ax_cdf):
            ax.axvline(x, color="#A1A1A6", linestyle="--", linewidth=1.5, alpha=0.8)
        self.ax_cdf.axhline(cdf_val, color="#A1A1A6", linestyle="--", linewidth=1.5, alpha=0.8)
        self.ax_cdf.plot([x], [cdf_val], "o", color="#FF453A", markersize=8)


# ---------------------------------------------------------------------------
# Parameter editor (rebuilt when distribution changes)
# ---------------------------------------------------------------------------

class ParamEditor(QWidget):
    valueChanged = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.form = QFormLayout(self)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.editors: dict[str, QDoubleSpinBox | QSpinBox] = {}
        self.sliders: dict[str, LinkedSlider] = {}

    @staticmethod
    def _slider_range_for_param(p: ParamSpec) -> tuple[float, float]:
        """Choose a useful slider range. Bounded if min/max are finite,
        otherwise heuristic around the default value."""
        default = float(p.default)
        lo_p = p.min_value if math.isfinite(p.min_value) else None
        hi_p = p.max_value if math.isfinite(p.max_value) else None
        if lo_p is not None and hi_p is not None:
            return lo_p, hi_p
        if lo_p is not None:
            span = max(abs(default - lo_p) * 3.0, 5.0)
            return lo_p, lo_p + span
        if hi_p is not None:
            span = max(abs(hi_p - default) * 3.0, 5.0)
            return hi_p - span, hi_p
        span = max(abs(default) * 3.0, 5.0)
        return default - span, default + span

    def set_params(self, params: list[ParamSpec]):
        while self.form.rowCount():
            self.form.removeRow(0)
        self.editors.clear()
        self.sliders.clear()
        for p in params:
            if p.integer:
                ed = QSpinBox()
                lo = int(max(-2_000_000_000, p.min_value)) if math.isfinite(p.min_value) else -2_000_000_000
                hi = int(min(2_000_000_000, p.max_value)) if math.isfinite(p.max_value) else 2_000_000_000
                ed.setRange(lo, hi)
                ed.setValue(int(p.default))
            else:
                ed = QDoubleSpinBox()
                ed.setDecimals(6)
                ed.setSingleStep(0.1)
                lo = p.min_value if math.isfinite(p.min_value) else -1e12
                hi = p.max_value if math.isfinite(p.max_value) else 1e12
                ed.setRange(lo, hi)
                ed.setValue(float(p.default))
            ed.setToolTip(p.description)
            ed.setMaximumWidth(110)
            slider_lo, slider_hi = self._slider_range_for_param(p)
            slider = LinkedSlider(ed, slider_lo, slider_hi)
            slider.valueChanged.connect(self.valueChanged.emit)
            label = QLabel(f"{p.name} — {p.description}")
            self.form.addRow(label, slider)
            self.editors[p.name] = ed
            self.sliders[p.name] = slider

    def values(self) -> dict[str, float]:
        return {name: ed.value() for name, ed in self.editors.items()}


# ---------------------------------------------------------------------------
# Operation panels
# ---------------------------------------------------------------------------

class ProbabilityPanel(QWidget):
    """Computes probabilities for various event types."""

    def __init__(self, main):
        super().__init__()
        self.main = main
        layout = QFormLayout(self)

        self.mode = QComboBox()
        self.mode.addItems([
            "P(X ≤ x)",
            "P(X < x)",
            "P(X ≥ x)",
            "P(X > x)",
            "P(X = x)",
            "P(a ≤ X ≤ b)",
        ])
        self.mode.currentIndexChanged.connect(self._on_mode_change)
        layout.addRow("Event:", self.mode)

        self.x_edit = QDoubleSpinBox()
        self.x_edit.setDecimals(6); self.x_edit.setRange(-1e12, 1e12); self.x_edit.setValue(0)
        self.x_edit.setMaximumWidth(110)
        self.x_slider = LinkedSlider(self.x_edit, -5, 5)
        self.x_slider.valueChanged.connect(self._on_value_change)
        layout.addRow("x:", self.x_slider)

        self.b_label = QLabel("b:")
        self.b_edit = QDoubleSpinBox()
        self.b_edit.setDecimals(6); self.b_edit.setRange(-1e12, 1e12); self.b_edit.setValue(1)
        self.b_edit.setMaximumWidth(110)
        self.b_slider = LinkedSlider(self.b_edit, -5, 5)
        self.b_slider.valueChanged.connect(self._on_value_change)
        layout.addRow(self.b_label, self.b_slider)

        self.live_check = QCheckBox("Live (compute while sliding)")
        self.live_check.setChecked(True)
        layout.addRow(self.live_check)

        btn = QPushButton("Compute")
        btn.clicked.connect(self.compute)
        layout.addRow(btn)

        self.result = QLabel("—")
        self.result.setStyleSheet("color: #0A84FF; font-size: 24px; font-weight: bold;")
        layout.addRow("Result:", self.result)

        self._on_mode_change()

    def _on_mode_change(self):
        is_between = self.mode.currentText() == "P(a ≤ X ≤ b)"
        self.b_slider.setVisible(is_between)
        self.b_label.setVisible(is_between)
        if self.live_check.isChecked():
            self.compute()

    def _on_value_change(self):
        if self.live_check.isChecked():
            self.compute()

    def update_slider_range(self, lo: float, hi: float):
        self.x_slider.set_range(lo, hi)
        self.b_slider.set_range(lo, hi)

    def compute(self):
        if not hasattr(self.main, "_current_dist"):
            return
        try:
            rv = self.main.current_rv()
            dist = self.main.current_dist
            x = self.x_edit.value()
            b = self.b_edit.value()
            mode = self.mode.currentText()
            highlight = None
            if mode == "P(X ≤ x)":
                p = float(rv.cdf(x)); highlight = {"mode": "le", "x": x}
            elif mode == "P(X < x)":
                if dist.discrete:
                    p = float(rv.cdf(x - 1))
                else:
                    p = float(rv.cdf(x))
                highlight = {"mode": "lt", "x": x}
            elif mode == "P(X ≥ x)":
                if dist.discrete:
                    p = 1.0 - float(rv.cdf(x - 1))
                else:
                    p = 1.0 - float(rv.cdf(x))
                highlight = {"mode": "ge", "x": x}
            elif mode == "P(X > x)":
                p = 1.0 - float(rv.cdf(x)); highlight = {"mode": "ge", "x": x + (1 if dist.discrete else 0)}
            elif mode == "P(X = x)":
                if dist.discrete:
                    p = float(rv.pmf(x))
                else:
                    p = 0.0
                highlight = {"mode": "le", "x": x}
            else:  # between
                a = x
                if dist.discrete:
                    p = float(rv.cdf(b) - rv.cdf(a - 1))
                else:
                    p = float(rv.cdf(b) - rv.cdf(a))
                highlight = {"mode": "between", "a": a, "b": b}
            self.result.setText(f"{p:.6f}")
            self.main.update_plot(highlight)
        except Exception as e:
            self.main.show_error(e)


class QuantilePanel(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        layout = QFormLayout(self)

        self.q_edit = QDoubleSpinBox()
        self.q_edit.setDecimals(6); self.q_edit.setRange(0.0, 1.0); self.q_edit.setValue(0.95)
        self.q_edit.setSingleStep(0.01)
        self.q_edit.setMaximumWidth(110)
        self.q_slider = LinkedSlider(self.q_edit, 0.0, 1.0)
        self.q_slider.valueChanged.connect(self._on_value_change)
        layout.addRow("q (Probability):", self.q_slider)

        self.live_check = QCheckBox("Live (compute while sliding)")
        self.live_check.setChecked(True)
        layout.addRow(self.live_check)

        btn = QPushButton("Compute Quantile")
        btn.clicked.connect(self.compute)
        layout.addRow(btn)

        self.result = QLabel("—")
        self.result.setStyleSheet("color: #0A84FF; font-size: 24px; font-weight: bold;")
        layout.addRow("x:", self.result)
        layout.addRow(QLabel("→ smallest x with P(X ≤ x) ≥ q"))

    def _on_value_change(self):
        if self.live_check.isChecked():
            self.compute()

    def compute(self):
        if not hasattr(self.main, "_current_dist"):
            return
        try:
            rv = self.main.current_rv()
            q = self.q_edit.value()
            x = float(rv.ppf(q))
            if self.main.current_dist.discrete:
                self.result.setText(f"{int(x)}")
            else:
                self.result.setText(f"{x:.6f}")
            self.main.update_plot({"mode": "le", "x": x})
        except Exception as e:
            self.main.show_error(e)


class SamplePanel(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.n_edit = QSpinBox(); self.n_edit.setRange(1, 100000); self.n_edit.setValue(20)
        form.addRow("Sample size n:", self.n_edit)
        self.seed_edit = QSpinBox(); self.seed_edit.setRange(0, 2_000_000_000); self.seed_edit.setValue(0)
        self.seed_edit.setSpecialValueText("random")
        form.addRow("Seed (0 = random):", self.seed_edit)
        layout.addLayout(form)

        btn = QPushButton("Draw Sample")
        btn.clicked.connect(self.compute)
        layout.addWidget(btn)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMaximumHeight(180)
        layout.addWidget(self.output)

        stats_form = QFormLayout()
        self.stats_label = QLabel("—")
        stats_form.addRow("Mean / SD / Min / Max:", self.stats_label)
        layout.addLayout(stats_form)

    def compute(self):
        try:
            rv = self.main.current_rv()
            n = self.n_edit.value()
            seed = self.seed_edit.value() or None
            samples = np.atleast_1d(rv.rvs(size=n, random_state=seed))
            if self.main.current_dist.discrete:
                lines = ", ".join(str(int(s)) for s in samples)
            else:
                lines = ", ".join(f"{s:.4f}" for s in samples)
            self.output.setPlainText(lines)
            self.stats_label.setText(
                f"{samples.mean():.4f}  /  {samples.std(ddof=1):.4f}  /  "
                f"{samples.min():.4f}  /  {samples.max():.4f}"
            )
            self.main.update_plot({"mode": "sample", "samples": samples})
        except Exception as e:
            self.main.show_error(e)


class SolvePanel(QWidget):
    """Solve for an unknown parameter given P(X ≤ x) = p."""

    def __init__(self, main):
        super().__init__()
        self.main = main
        layout = QFormLayout(self)

        self.param_box = QComboBox()
        layout.addRow("Unknown Parameter:", self.param_box)

        self.x_edit = QDoubleSpinBox()
        self.x_edit.setDecimals(6); self.x_edit.setRange(-1e12, 1e12); self.x_edit.setValue(0)
        layout.addRow("x:", self.x_edit)

        self.p_edit = QDoubleSpinBox()
        self.p_edit.setDecimals(6); self.p_edit.setRange(0.0, 1.0); self.p_edit.setValue(0.95)
        self.p_edit.setSingleStep(0.01)
        layout.addRow("P(X ≤ x):", self.p_edit)

        self.lo_edit = QDoubleSpinBox()
        self.lo_edit.setDecimals(6); self.lo_edit.setRange(-1e12, 1e12); self.lo_edit.setValue(0.001)
        layout.addRow("Search Range min:", self.lo_edit)
        self.hi_edit = QDoubleSpinBox()
        self.hi_edit.setDecimals(6); self.hi_edit.setRange(-1e12, 1e12); self.hi_edit.setValue(100.0)
        layout.addRow("Search Range max:", self.hi_edit)

        btn = QPushButton("Solve Parameter")
        btn.clicked.connect(self.compute)
        layout.addRow(btn)

        self.result = QLabel("—")
        self.result.setStyleSheet("color: #0A84FF; font-size: 24px; font-weight: bold;")
        layout.addRow("Result:", self.result)

    def refresh_params(self, dist: Distribution):
        self.param_box.clear()
        self.param_box.addItems([p.name for p in dist.params])

    def compute(self):
        try:
            dist = self.main.current_dist
            if not dist.params:
                self.result.setText("(no parameters)")
                return
            unknown = self.param_box.currentText()
            x = self.x_edit.value()
            target_p = self.p_edit.value()
            base_values = self.main.param_editor.values()

            def f(theta):
                vals = dict(base_values)
                vals[unknown] = theta
                try:
                    rv = dist.build(vals, enforce_integer=False)
                    return float(rv.cdf(x)) - target_p
                except Exception:
                    return float("nan")

            lo, hi = self.lo_edit.value(), self.hi_edit.value()
            f_lo, f_hi = f(lo), f(hi)

            # Detect distributions where scipy needs strict integer params
            # → integer grid search instead of brentq
            param_spec = next(p for p in dist.params if p.name == unknown)
            mid_test = f((lo + hi) / 2)
            need_int_search = param_spec.integer and math.isnan(mid_test)

            if need_int_search:
                lo_i, hi_i = int(math.ceil(lo)), int(math.floor(hi))
                best, best_err = None, math.inf
                for k in range(lo_i, hi_i + 1):
                    err = f(float(k))
                    if not math.isnan(err) and abs(err) < best_err:
                        best, best_err = k, abs(err)
                if best is None:
                    raise ValueError("No solution found in search range.")
                root = float(best)
                self.result.setText(f"{unknown} = {best}  (Δ={best_err:.4g})")
            else:
                if math.isnan(f_lo) or math.isnan(f_hi):
                    raise ValueError("Search range contains invalid values — adjust bounds.")
                if f_lo * f_hi > 0:
                    raise ValueError(
                        "Solution not in search range (identical signs). "
                        "Expand bounds."
                    )
                root = optimize.brentq(f, lo, hi, xtol=1e-9)
                self.result.setText(f"{unknown} = {root:.6f}")

            vals = dict(base_values); vals[unknown] = root
            rv = dist.build(vals, enforce_integer=False)
            self.main.canvas.plot(dist, rv, {"mode": "le", "x": x})
        except Exception as e:
            self.main.show_error(e)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("StatiQ — Statistics Studio")
        self.resize(1280, 800)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # --- Left: search and distribution list ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Filter distributions...")
        self.search_bar.textChanged.connect(self._on_search)
        left_layout.addWidget(self.search_bar)

        left = QTabWidget()
        self.discrete_list = QListWidget()
        for d in DISCRETE:
            QListWidgetItem(d.name, self.discrete_list)
        self.continuous_list = QListWidget()
        for d in CONTINUOUS:
            QListWidgetItem(d.name, self.continuous_list)
        left.addTab(self.discrete_list, f"Discrete ({len(DISCRETE)})")
        left.addTab(self.continuous_list, f"Continuous ({len(CONTINUOUS)})")
        self.discrete_list.currentItemChanged.connect(self._on_select)
        self.continuous_list.currentItemChanged.connect(self._on_select)
        left.currentChanged.connect(self._on_tab_change)
        left_layout.addWidget(left)

        splitter.addWidget(left_widget)

        # --- Center: parameters + operations ---
        center = QWidget()
        center_layout = QVBoxLayout(center)

        # --- Info Box & Learning Mode ---
        self.learning_mode_check = QCheckBox("📖 Learning Mode")
        self.learning_mode_check.setStyleSheet("font-weight: bold; color: #0A84FF; margin-bottom: 5px;")
        self.learning_mode_check.setChecked(True)
        self.learning_mode_check.stateChanged.connect(self._toggle_learning_mode)
        center_layout.addWidget(self.learning_mode_check)

        self.info_box = QLabel("")
        self.info_box.setWordWrap(True)
        self.info_box.setStyleSheet("color: #E0E0E0; font-size: 13px; margin-bottom: 10px; background-color: #2D2D30; padding: 12px; border-radius: 6px; border-left: 4px solid #0A84FF;")
        self.info_box.setVisible(False)
        center_layout.addWidget(self.info_box)

        param_box = QGroupBox("Parameters")
        pl = QVBoxLayout(param_box)
        self.param_editor = ParamEditor()
        self.param_editor.valueChanged.connect(self._on_param_change)
        pl.addWidget(self.param_editor)
        center_layout.addWidget(param_box)

        op_box = QGroupBox("Operations")
        ol = QVBoxLayout(op_box)
        self.op_tabs = QTabWidget()
        self.prob_panel = ProbabilityPanel(self)
        self.quant_panel = QuantilePanel(self)
        self.sample_panel = SamplePanel(self)
        self.solve_panel = SolvePanel(self)
        self.op_tabs.addTab(self.prob_panel, "Probability")
        self.op_tabs.addTab(self.quant_panel, "Quantile")
        self.op_tabs.addTab(self.sample_panel, "Random Sample")
        self.op_tabs.addTab(self.solve_panel, "Solve Parameter")
        ol.addWidget(self.op_tabs)
        center_layout.addWidget(op_box, 1)

        splitter.addWidget(center)

        # --- Right: plots ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.canvas = PlotCanvas()
        right_layout.addWidget(self.canvas)
        
        export_btn = QPushButton("Export Plot (PNG)")
        export_btn.clicked.connect(self._export_plot)
        right_layout.addWidget(export_btn)

        splitter.addWidget(right_widget)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([220, 380, 680])

        self._build_menu()

        # Default selection
        self.continuous_list.setCurrentRow(0)  # Normal
        left.setCurrentIndex(1)

    def _build_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        quit_act = QAction("Exit", self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        help_menu = menubar.addMenu("&Help")
        about_act = QAction("About", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    def _show_about(self):
        QMessageBox.information(
            self, "About StatiQ",
            "StatiQ — Statistik Studio\n\n"
            "Interactive tool for probabilities, quantiles, and sampling.\n"
            "Based on PyQRS, completely overhauled for learning and accessibility."
        )

    def _toggle_learning_mode(self):
        if hasattr(self, '_current_dist') and self.learning_mode_check.isChecked() and self._current_dist.info_text:
            self.info_box.setVisible(True)
        else:
            self.info_box.setVisible(False)

    def _export_plot(self):
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self, "Save Plot", "StatiQ_Plot.png", "Images (*.png)")
        if path:
            self.canvas.fig.savefig(path, facecolor=self.canvas.fig.get_facecolor(), edgecolor='none', dpi=300)
            QMessageBox.information(self, "Exportiert", f"Plot successfully saved to {path}.")

    def _on_search(self, text: str):
        text = text.lower()
        for i in range(self.discrete_list.count()):
            item = self.discrete_list.item(i)
            item.setHidden(text not in item.text().lower())
        for i in range(self.continuous_list.count()):
            item = self.continuous_list.item(i)
            item.setHidden(text not in item.text().lower())

    @property
    def current_dist(self) -> Distribution:
        return self._current_dist

    def current_rv(self):
        return self.current_dist.build(self.param_editor.values())

    def _on_tab_change(self, idx: int):
        lst = self.discrete_list if idx == 0 else self.continuous_list
        if lst.currentRow() < 0:
            lst.setCurrentRow(0)
        else:
            self._on_select(lst.currentItem(), None)

    def _on_select(self, current: QListWidgetItem | None, _previous):
        if current is None:
            return
        name = current.text()
        from distributions import BY_NAME
        dist = BY_NAME[name]
        self._current_dist = dist
        self.param_editor.set_params(dist.params)
        self.solve_panel.refresh_params(dist)
        if dist.info_text:
            self.info_box.setText(dist.info_text)
            self.info_box.setVisible(self.learning_mode_check.isChecked())
        else:
            self.info_box.setVisible(False)
        # Match x/b sliders to current distribution support
        try:
            rv = self.current_rv()
            lo, hi = dist.support_range(rv)
        except Exception:
            lo, hi = -5.0, 5.0
        self.prob_panel.update_slider_range(lo, hi)
        self.prob_panel.x_edit.setValue(dist.default_x)
        self.prob_panel.b_edit.setValue(dist.default_x + max(1.0, (hi - lo) * 0.2))
        self.update_plot(None)
        if self.op_tabs.currentWidget() is self.prob_panel and self.prob_panel.live_check.isChecked():
            self.prob_panel.compute()
        elif self.op_tabs.currentWidget() is self.quant_panel and self.quant_panel.live_check.isChecked():
            self.quant_panel.compute()

    def _on_param_change(self):
        try:
            rv = self.current_rv()
            lo, hi = self.current_dist.support_range(rv)
            self.prob_panel.update_slider_range(lo, hi)
        except Exception:
            pass
        # Re-run whichever operation is currently visible (if live)
        current = self.op_tabs.currentWidget()
        if current is self.prob_panel and self.prob_panel.live_check.isChecked():
            self.prob_panel.compute()
        elif current is self.quant_panel and self.quant_panel.live_check.isChecked():
            self.quant_panel.compute()
        else:
            self.update_plot(None)

    def update_plot(self, highlight: dict | None):
        try:
            rv = self.current_rv()
            self.canvas.plot(self.current_dist, rv, highlight)
        except Exception as e:
            self.show_error(e)

    def show_error(self, e: Exception):
        import traceback
        print(f"[PyQRS error] {e!r}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        msg = str(e) or repr(e)
        QMessageBox.warning(self, "Error", msg)


def main():
    import os
    app = QApplication(sys.argv)
    app.setApplicationName("StatiQ")
    
    style_path = os.path.join(os.path.dirname(__file__), "style.qss")
    if os.path.exists(style_path):
        with open(style_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
            
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
