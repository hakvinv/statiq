"""Quick calc command bar + distribution table (PQRS-style, result first).

Grammar:
    <dist> <op> <value> [params]      e.g.  norm cdf 1.96 | t quant .975 df=20
    <dist> P(X<=v) [params]           e.g.  binom P(X<=3) n=10 p=.5
    <dist> table [params]             e.g.  chi2 table df=5
Ops: cdf pdf pmf sf quant (aliases ppf/inv). Missing params fall back to defaults.
"""
from __future__ import annotations

import math

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView, QDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

ALIASES = {
    "norm": "Normal", "normal": "Normal",
    "z": "Standard Normal", "stdnorm": "Standard Normal", "snorm": "Standard Normal",
    "t": "Student-t", "student": "Student-t", "studentt": "Student-t",
    "chi2": "Chi-Square", "chisq": "Chi-Square", "chisquare": "Chi-Square",
    "ncx2": "Noncentral Chi-Square", "ncchi2": "Noncentral Chi-Square",
    "f": "F-Distribution", "fish": "F-Distribution", "fisher": "F-Distribution",
    "ncf": "Noncentral F",
    "wilcoxon": "Wilcoxon Signed-Rank", "wsr": "Wilcoxon Signed-Rank",
    "mwu": "Mann-Whitney U", "mannwhitney": "Mann-Whitney U",
    "cauchy": "Cauchy", "logistic": "Logistic", "laplace": "Laplace",
    "pareto": "Pareto", "triangular": "Triangular", "triang": "Triangular",
    "gumbel": "Gumbel", "rayleigh": "Rayleigh",
    "invgauss": "Inverse Gaussian", "inversegaussian": "Inverse Gaussian",
    "logseries": "Logarithmic (Log-Series)", "logarithmic": "Logarithmic (Log-Series)",
    "binom": "Binomial", "binomial": "Binomial",
    "pois": "Poisson", "poisson": "Poisson",
    "exp": "Exponential", "exponential": "Exponential",
    "geom": "Geometric",
    "nbinom": "Negative Binomial", "negbin": "Negative Binomial",
    "hyper": "Hypergeometric", "hypergeom": "Hypergeometric",
    "dunif": "Discrete Uniform", "discunif": "Discrete Uniform",
    "unif": "Continuous Uniform", "uniform": "Continuous Uniform",
    "beta": "Beta", "gamma": "Gamma",
    "lognorm": "Lognormal", "lognormal": "Lognormal",
    "weibull": "Weibull", "weib": "Weibull",
    "bern": "Bernoulli", "bernoulli": "Bernoulli",
}

OP_ALIASES = {
    "cdf": "cdf", "prob": "cdf", "p": "cdf",
    "pdf": "pdf", "density": "pdf", "d": "pdf",
    "pmf": "pmf", "mass": "pmf",
    "sf": "sf", "survival": "sf", "ccdf": "sf",
    "quant": "quant", "quantile": "quant", "ppf": "quant", "inv": "quant", "q": "quant",
}

PARAM_ALIASES = {
    "mu": "μ", "mean": "μ", "m": "μ",
    "sigma": "σ", "sd": "σ", "s": "σ",
    "lambda": "λ", "lam": "λ", "rate": "λ", "l": "λ",
    "alpha": "α", "beta": "β", "df1": "df₁", "df2": "df₂",
    "theta": "θ", "scale": "θ", "shape": "k",
}

SYM = {"le": "≤", "lt": "<", "ge": "≥", "gt": ">", "eq": "="}


class CommandError(ValueError):
    pass


def _norm_param(raw: str) -> str:
    key = raw.lower().rstrip("=:")
    return PARAM_ALIASES.get(key, raw)


def _find_param(dist, key: str):
    wanted = _norm_param(key)
    for p in dist.params:
        if p.name == wanted:
            return p
    low = wanted.lower()
    for p in dist.params:
        if p.name.lower() == low:
            return p
    return None


def parse_command(raw: str, by_name: dict):
    text = " ".join(str(raw or "").split())
    if not text:
        raise CommandError("Empty command.")
    table = False
    event = None
    import re
    em = re.search(r"p\s*\(\s*x\s*(<=|>=|<|>|=)\s*(-?[\d.]+)\s*\)", text, re.I)
    if em:
        cmp_ = em.group(1)
        event = {"cmp": cmp_, "x": float(em.group(2)),
                 "mode": {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "=": "eq"}[cmp_]}
        text = (text[: em.start()] + " " + text[em.end():]).strip()
        text = " ".join(text.split())
    tokens = text.split()
    if not tokens:
        raise CommandError("Missing distribution.")
    alias = tokens.pop(0).lower()
    dist_name = ALIASES.get(alias)
    if not dist_name:
        raise CommandError(f'Unknown distribution "{alias}". Try: norm, t, chi2, f, binom, pois, exp, beta, gamma, unif, ...')
    dist = by_name[dist_name]
    if tokens and tokens[0].lower() == "table":
        table = True
        tokens.pop(0)

    op = None
    if tokens and tokens[0].lower() in OP_ALIASES:
        op = OP_ALIASES[tokens.pop(0).lower()]

    value = None
    if tokens and "=" not in tokens[0]:
        try:
            value = float(tokens[0])
            tokens.pop(0)
        except ValueError:
            pass

    params = {}
    while tokens:
        key = tokens.pop(0)
        if "=" in key:
            key, _, rawval = key.partition("=")
            try:
                val = float(rawval)
            except ValueError:
                raise CommandError(f'Bad value for parameter "{key}".')
        else:
            if not tokens:
                raise CommandError(f'Missing value for parameter "{key}".')
            try:
                val = float(tokens.pop(0))
            except ValueError:
                raise CommandError(f'Missing value for parameter "{key}".')
        spec = _find_param(dist, key)
        if not spec:
            available = ", ".join(p.name for p in dist.params) or "none"
            raise CommandError(f'"{dist.name}" has no parameter "{key}". Available: {available}.')
        params[spec.name] = int(round(val)) if spec.integer else val

    if event:
        if value is not None:
            raise CommandError("Value given twice (P(X≤v) and bare number).")
        value = event["x"]
        if not op:
            op = "pmf" if event["cmp"] == "=" else ("cdf" if event["cmp"] in ("<", "<=") else "sf")
    if table:
        return {"dist_name": dist_name, "dist": dist, "table": True, "params": params,
                "op": None, "value": None, "event": None}
    if not op:
        raise CommandError("Missing operation: cdf | pdf | pmf | sf | quant — e.g. norm cdf 1.96")
    if op == "pdf" and dist.discrete:
        op = "pmf"
    if op == "pmf" and not dist.discrete:
        op = "pdf"
    if value is None:
        raise CommandError("Missing value, e.g. norm cdf 1.96")
    return {"dist_name": dist_name, "dist": dist, "table": False, "op": op,
            "value": value, "params": params, "event": event}


def merged_params(dist, given: dict):
    merged, defaulted = {}, []
    for p in dist.params:
        if p.name in given:
            merged[p.name] = given[p.name]
        else:
            merged[p.name] = p.default
            defaulted.append(p.name)
    return merged, defaulted


def compute_command(cmd: dict):
    dist, x = cmd["dist"], cmd["value"]
    merged, _ = merged_params(dist, cmd["params"])
    rv = dist.build(merged)
    op = cmd["op"]
    if op == "quant":
        if not 0 < x < 1:
            raise CommandError("Quantile needs q in (0,1), e.g. t quant 0.975 df=20")
        qx = float(rv.ppf(x))
        return f"x = {int(round(qx)) if dist.discrete else f'{qx:.6f}'}", f"P(X ≤ x) = {x}"
    if op == "cdf":
        mode = cmd["event"]["mode"] if cmd["event"] else "le"
        if mode == "le":
            p = float(rv.cdf(x))
        elif mode == "lt":
            p = float(rv.cdf(x - 1)) if dist.discrete else float(rv.cdf(x))
        elif mode == "ge":
            p = 1 - (float(rv.cdf(x - 1)) if dist.discrete else float(rv.cdf(x)))
        elif mode == "gt":
            p = 1 - float(rv.cdf(x))
        else:
            p = float(rv.pmf(x)) if dist.discrete else 0.0
        sym = SYM[mode] if cmd["event"] else "≤"
        return f"P(X {sym} {x}) = {p:.6f}", ""
    if op == "sf":
        if cmd["event"]:
            mode = cmd["event"]["mode"]
            if mode == "ge":
                p = 1 - (float(rv.cdf(x - 1)) if dist.discrete else float(rv.cdf(x)))
                return f"P(X ≥ {x}) = {p:.6f}", ""
            # gt
            return f"P(X > {x}) = {1 - float(rv.cdf(x)):.6f}", ""
        return f"P(X > {x}) = {1 - float(rv.cdf(x)):.6f}", ""
    if op == "pdf":
        return f"f({x}) = {float(rv.pdf(x)):.6f}", ""
    if op == "pmf":
        return f"P(X = {x}) = {float(rv.pmf(x)):.6f}", ""
    raise CommandError("Unsupported operation.")


def describe_params(dist, given: dict) -> str:
    merged, defaulted = merged_params(dist, given)
    if not dist.params:
        return ""
    parts = ", ".join(f"{p.name}={merged[p.name]}" for p in dist.params)
    note = f" · default for {', '.join(defaulted)}" if defaulted else ""
    return f" ({parts}){note}"


class TableDialog(QDialog):
    def __init__(self, parent, dist, params: dict):
        super().__init__(parent)
        merged, _ = merged_params(dist, params)
        rv = dist.build(merged)
        lo, hi = dist.support_range(rv)
        self.setWindowTitle(f"{dist.name} — table{describe_params(dist, params)}")
        self.resize(420, 560)
        layout = QVBoxLayout(self)
        table = QTableWidget(self)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["x", "P(X = x)" if dist.discrete else "f(x)", "F(x) = P(X ≤ x)"])
        if dist.discrete:
            xs = list(range(math.floor(lo), math.ceil(hi) + 1))
        else:
            n = 49
            xs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
        table.setRowCount(len(xs))
        for row, xv in enumerate(xs):
            if dist.discrete:
                fx = float(rv.pmf(int(xv)))
                vals = (f"{int(xv)}", f"{fx:.6f}", f"{float(rv.cdf(int(xv))):.6f}")
            else:
                vals = (f"{xv:.2f}", f"{float(rv.pdf(xv)):.6f}", f"{float(rv.cdf(xv)):.6f}")
            for col, val in enumerate(vals):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(row, col, item)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(table)


class QuickCalcBar(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window = window
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText(
            "norm cdf 1.96  ·  binom P(X<=3) n=10 p=.5  ·  t quant .975 df=20  ·  chi2 table df=5")
        self.input.returnPressed.connect(self.run)
        run_btn = QPushButton("Run")
        run_btn.clicked.connect(self.run)
        row.addWidget(self.input, 1)
        row.addWidget(run_btn)
        layout.addLayout(row)
        self.result = QLabel("Quick calc — number first. Press / to focus.")
        self.result.setStyleSheet("color: #9A9AA0; font-size: 14px; padding: 4px 2px;")
        self.result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.result)

    def run(self):
        from distributions import BY_NAME
        try:
            cmd = parse_command(self.input.text(), BY_NAME)
        except CommandError as e:
            self._show(str(e), error=True)
            return
        try:
            if cmd["table"]:
                TableDialog(self.window, cmd["dist"], cmd["params"]).exec()
                self.window.apply_command(cmd, select_only=True)
                self._show(f"Table: {cmd['dist_name']}{describe_params(cmd['dist'], cmd['params'])}")
                return
            text, detail = compute_command(cmd)
            self.window.apply_command(cmd)
            suffix = describe_params(cmd["dist"], cmd["params"])
            shown = text + (f"   {detail}" if detail else "") + suffix
            self._show(shown)
        except Exception as e:
            self._show(str(e) or repr(e), error=True)

    def _show(self, text: str, error: bool = False):
        color = "#FF6961" if error else "#0A84FF"
        size = "14px" if error else "22px"
        self.result.setStyleSheet(f"color: {color}; font-size: {size}; font-weight: bold; padding: 4px 2px;")
        self.result.setText(text)
