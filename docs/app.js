// statiq — browser app powered by Pyodide

let pyodide;
let DISTS = [];           // list of distribution metadata
let currentDist = null;   // currently selected distribution
let currentParams = {};   // {paramName: value}
let currentOp = "probability";
const SLIDER_STEPS = 1000;

const loadingEl = document.getElementById("loading");
const loadingSubEl = document.getElementById("loading-sub");
const uiEl = document.getElementById("ui");

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

const t0 = Date.now();
function elapsed() { return ((Date.now() - t0) / 1000).toFixed(1) + "s"; }
function setStatus(msg) {
  console.log(`[statiq +${elapsed()}]`, msg);
  loadingSubEl.textContent = `${msg}  (${elapsed()})`;
}

async function init() {
  try {
    if (typeof loadPyodide !== "function") {
      throw new Error("loadPyodide not available — pyodide.js CDN failed to load");
    }
    setStatus("loading Pyodide runtime (≈10 MB)…");
    pyodide = await loadPyodide({
      stdout: (s) => console.log("[py]", s),
      stderr: (s) => console.error("[py]", s),
    });
    setStatus("Pyodide ready. Loading numpy + scipy (≈50 MB; this is the slow part)…");
    await pyodide.loadPackage(["numpy", "scipy"], {
      messageCallback: (m) => setStatus(`${m}`),
      errorCallback: (m) => console.error("[load]", m),
    });

    setStatus("Loading statiq distribution module…");
    const distRes = await fetch("distributions.py");
    if (!distRes.ok) throw new Error(`distributions.py: HTTP ${distRes.status}`);
    const distCode = await distRes.text();
    pyodide.runPython(distCode);

    // Define helper functions in Python
    pyodide.runPython(`
import json, math, numpy as np
from scipy import optimize

def list_distributions():
    out = []
    for d in ALL_DISTRIBUTIONS:
        out.append({
            "name": d.name,
            "discrete": d.discrete,
            "default_x": d.default_x,
            "info_text": d.info_text,
            "category": d.category,
            "params": [
                {
                    "name": p.name,
                    "description": p.description,
                    "default": float(p.default),
                    "integer": p.integer,
                    "min_value": None if not math.isfinite(p.min_value) else p.min_value,
                    "max_value": None if not math.isfinite(p.max_value) else p.max_value,
                    "exclusive_min": p.exclusive_min,
                    "exclusive_max": p.exclusive_max,
                }
                for p in d.params
            ],
        })
    return json.dumps(out)


def _build_rv(dist_name, params):
    dist = BY_NAME[dist_name]
    return dist, dist.build(dict(params), enforce_integer=False)


def get_plot_data(dist_name, params):
    dist, rv = _build_rv(dist_name, params)
    try:
        lo, hi = dist.support_range(rv)
    except Exception:
        lo, hi = -5.0, 5.0
    if hi <= lo:
        hi = lo + 1.0
    if dist.discrete:
        k_lo = int(math.floor(lo)) - 1
        k_hi = int(math.ceil(hi)) + 1
        ks = list(range(k_lo, k_hi + 1))
        pdf = [float(v) for v in rv.pmf(np.asarray(ks))]
        cdf = [float(v) for v in rv.cdf(np.asarray(ks))]
    else:
        xs = np.linspace(lo, hi, 400)
        pdf = [float(v) for v in rv.pdf(xs)]
        cdf = [float(v) for v in rv.cdf(xs)]
        ks = [float(v) for v in xs]
    return json.dumps({"discrete": dist.discrete, "x": ks, "pdf": pdf, "cdf": cdf, "lo": float(lo), "hi": float(hi)})


def compute_probability(dist_name, params, mode, x, b=None):
    dist, rv = _build_rv(dist_name, params)
    if mode == "le":
        p = float(rv.cdf(x))
    elif mode == "lt":
        p = float(rv.cdf(x - 1)) if dist.discrete else float(rv.cdf(x))
    elif mode == "ge":
        p = 1.0 - (float(rv.cdf(x - 1)) if dist.discrete else float(rv.cdf(x)))
    elif mode == "gt":
        p = 1.0 - float(rv.cdf(x))
    elif mode == "eq":
        p = float(rv.pmf(x)) if dist.discrete else 0.0
    elif mode == "between":
        if dist.discrete:
            p = float(rv.cdf(b) - rv.cdf(x - 1))
        else:
            p = float(rv.cdf(b) - rv.cdf(x))
    else:
        p = float("nan")
    return p


def compute_quantile(dist_name, params, q):
    dist, rv = _build_rv(dist_name, params)
    return float(rv.ppf(q))


def draw_sample(dist_name, params, n, seed):
    dist, rv = _build_rv(dist_name, params)
    s = np.atleast_1d(rv.rvs(size=n, random_state=(seed or None)))
    if dist.discrete:
        values = [int(v) for v in s]
    else:
        values = [float(v) for v in s]
    arr = np.asarray(values, dtype=float)
    return json.dumps({
        "values": values,
        "discrete": dist.discrete,
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    })


def solve_parameter(dist_name, params, unknown_param, x, target_p, lo, hi):
    dist = BY_NAME[dist_name]
    base = dict(params)
    param_spec = next(p for p in dist.params if p.name == unknown_param)

    def f(theta):
        vals = dict(base)
        vals[unknown_param] = theta
        try:
            rv = dist.build(vals, enforce_integer=False)
            return float(rv.cdf(x)) - target_p
        except Exception:
            return float("nan")

    mid_test = f((lo + hi) / 2)
    need_int = param_spec.integer and math.isnan(mid_test)
    if need_int:
        lo_i, hi_i = int(math.ceil(lo)), int(math.floor(hi))
        best, best_err = None, math.inf
        for k in range(lo_i, hi_i + 1):
            err = f(float(k))
            if not math.isnan(err) and abs(err) < best_err:
                best, best_err = k, abs(err)
        if best is None:
            raise ValueError("Keine Lösung im Suchbereich gefunden.")
        return json.dumps({"value": float(best), "integer_search": True, "error": float(best_err)})
    else:
        f_lo, f_hi = f(lo), f(hi)
        if math.isnan(f_lo) or math.isnan(f_hi):
            raise ValueError("Suchbereich enthält ungültige Werte.")
        if f_lo * f_hi > 0:
            raise ValueError("Lösung liegt nicht im Suchbereich (Vorzeichen identisch).")
        root = optimize.brentq(f, lo, hi, xtol=1e-9)
        return json.dumps({"value": float(root), "integer_search": False})
`);

    setStatus("Building distribution metadata…");
    const distsJson = pyodide.runPython("list_distributions()");
    DISTS = JSON.parse(distsJson);
    setStatus(`Ready — ${DISTS.length} distributions loaded.`);

    setupUI();
    loadingEl.hidden = true;
    uiEl.hidden = false;
  } catch (e) {
    const detail = (e && e.message) ? e.message : String(e);
    loadingSubEl.innerHTML = `<div style="color:#ff453a; max-width:600px; white-space:pre-wrap; text-align:left; font-family:monospace; font-size:0.8rem;">Failed: ${detail}</div>`;
    console.error("[statiq init failed]", e);
  }
}

// ---------------------------------------------------------------------------
// UI Setup
// ---------------------------------------------------------------------------

function setupUI() {
  const select = document.getElementById("dist-select");
  // group by discrete/continuous
  const discreteOptgroup = document.createElement("optgroup");
  discreteOptgroup.label = `Discrete (${DISTS.filter(d => d.discrete).length})`;
  const continuousOptgroup = document.createElement("optgroup");
  continuousOptgroup.label = `Continuous (${DISTS.filter(d => !d.discrete).length})`;
  for (const d of DISTS) {
    const opt = document.createElement("option");
    opt.value = d.name;
    opt.textContent = d.name;
    (d.discrete ? discreteOptgroup : continuousOptgroup).appendChild(opt);
  }
  select.appendChild(discreteOptgroup);
  select.appendChild(continuousOptgroup);
  select.addEventListener("change", () => selectDistribution(select.value));

  // Tab buttons
  document.querySelectorAll(".tab").forEach(btn => {
    btn.addEventListener("click", () => switchOp(btn.dataset.op));
  });

  // Probability panel
  document.getElementById("prob-mode").addEventListener("change", () => {
    const mode = document.getElementById("prob-mode").value;
    document.getElementById("prob-b-row").hidden = mode !== "between";
    runProbability();
  });
  setupLinkedNumberSlider("prob-x", () => runProbability(), () => xRange());
  setupLinkedNumberSlider("prob-b", () => runProbability(), () => xRange());

  // Quantile panel
  setupLinkedNumberSlider("quant-q", () => runQuantile(), () => ({ lo: 0, hi: 1 }));

  // Sample panel
  document.getElementById("sample-btn").addEventListener("click", runSample);

  // Solve panel
  document.getElementById("solve-btn").addEventListener("click", runSolve);

  // Default selection
  selectDistribution("Normal");
}

function switchOp(op) {
  currentOp = op;
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.op === op));
  document.querySelectorAll(".op-panel").forEach(p => p.hidden = p.dataset.op !== op);
  redrawPlot();
  if (op === "probability") runProbability();
  else if (op === "quantile") runQuantile();
}

// ---------------------------------------------------------------------------
// Distribution change
// ---------------------------------------------------------------------------

function selectDistribution(name) {
  currentDist = DISTS.find(d => d.name === name);
  document.getElementById("dist-select").value = name;
  currentParams = {};
  for (const p of currentDist.params) currentParams[p.name] = p.default;

  document.getElementById("dist-info").innerHTML = currentDist.info_text || "";
  buildParamsUI();
  buildSolveParamSelect();

  // Set default x for probability
  setNumber("prob-x", currentDist.default_x);
  // Set b default = default_x + something
  setNumber("prob-b", currentDist.default_x + 1.0);

  // Update slider ranges that depend on support
  updateProbabilitySliderRange();

  redrawPlot();
  if (currentOp === "probability") runProbability();
  if (currentOp === "quantile") runQuantile();
}

function buildParamsUI() {
  const container = document.getElementById("params");
  container.innerHTML = "";
  if (currentDist.params.length === 0) {
    container.innerHTML = `<p class="hint">No parameters.</p>`;
    return;
  }
  for (const p of currentDist.params) {
    const row = document.createElement("div");
    row.className = "slider-row";
    row.innerHTML = `
      <span class="slider-label" title="${escapeAttr(p.description)}">${p.name}</span>
      <input type="number" id="param-num-${cssId(p.name)}" step="${p.integer ? 1 : "any"}">
      <input type="range" id="param-slider-${cssId(p.name)}" min="0" max="${SLIDER_STEPS}" value="500">
    `;
    container.appendChild(row);

    const numEl = document.getElementById(`param-num-${cssId(p.name)}`);
    const sliderEl = document.getElementById(`param-slider-${cssId(p.name)}`);
    numEl.value = p.default;

    const [lo, hi] = sliderRangeForParam(p);
    linkNumSlider(numEl, sliderEl, lo, hi, p.integer, (v) => {
      currentParams[p.name] = v;
      onParamChange();
    });
  }
}

function buildSolveParamSelect() {
  const sel = document.getElementById("solve-param");
  sel.innerHTML = "";
  for (const p of currentDist.params) {
    const opt = document.createElement("option");
    opt.value = p.name;
    opt.textContent = p.name;
    sel.appendChild(opt);
  }
}

function sliderRangeForParam(p) {
  const def = p.default;
  if (p.min_value !== null && p.max_value !== null) return [p.min_value, p.max_value];
  if (p.min_value !== null) {
    const span = Math.max(Math.abs(def - p.min_value) * 3, 5);
    return [p.min_value, p.min_value + span];
  }
  if (p.max_value !== null) {
    const span = Math.max(Math.abs(p.max_value - def) * 3, 5);
    return [p.max_value - span, p.max_value];
  }
  const span = Math.max(Math.abs(def) * 3, 5);
  return [def - span, def + span];
}

function onParamChange() {
  updateProbabilitySliderRange();
  redrawPlot();
  if (currentOp === "probability") runProbability();
  else if (currentOp === "quantile") runQuantile();
}

function updateProbabilitySliderRange() {
  const { lo, hi } = xRange();
  rescaleSlider("prob-x", lo, hi);
  rescaleSlider("prob-b", lo, hi);
}

function xRange() {
  // Use the current plot data range (cached)
  if (window._lastPlotData) {
    return { lo: window._lastPlotData.lo, hi: window._lastPlotData.hi };
  }
  return { lo: -5, hi: 5 };
}

// ---------------------------------------------------------------------------
// Linked number+slider helpers
// ---------------------------------------------------------------------------

function linkNumSlider(numEl, sliderEl, lo, hi, integer, onChange) {
  numEl._linkedRange = { lo, hi, integer };
  let syncing = false;
  function fromSlider() {
    if (syncing) return;
    const pos = +sliderEl.value;
    let v = numEl._linkedRange.lo + (pos / SLIDER_STEPS) * (numEl._linkedRange.hi - numEl._linkedRange.lo);
    if (numEl._linkedRange.integer) v = Math.round(v);
    syncing = true;
    numEl.value = numEl._linkedRange.integer ? v : +v.toFixed(6);
    syncing = false;
    onChange(+numEl.value);
  }
  function fromNum() {
    if (syncing) return;
    const v = +numEl.value;
    let pos = (v - numEl._linkedRange.lo) / (numEl._linkedRange.hi - numEl._linkedRange.lo) * SLIDER_STEPS;
    pos = Math.max(0, Math.min(SLIDER_STEPS, pos));
    syncing = true;
    sliderEl.value = Math.round(pos);
    syncing = false;
    onChange(v);
  }
  sliderEl.addEventListener("input", fromSlider);
  numEl.addEventListener("input", fromNum);
  // Sync initial slider position
  syncing = true;
  fromNum.call();
  syncing = false;
}

function setupLinkedNumberSlider(id, onChange, rangeFn) {
  const numEl = document.getElementById(`${id}-num`);
  const sliderEl = document.getElementById(`${id}-slider`);
  const { lo, hi } = rangeFn();
  linkNumSlider(numEl, sliderEl, lo, hi, false, onChange);
  // Store the rangeFn so we can rescale
  numEl._rangeFn = rangeFn;
  sliderEl._rangeFn = rangeFn;
}

function rescaleSlider(id, lo, hi) {
  const numEl = document.getElementById(`${id}-num`);
  if (!numEl) return;
  numEl._linkedRange.lo = lo;
  numEl._linkedRange.hi = hi;
  // Re-sync slider position
  const sliderEl = document.getElementById(`${id}-slider`);
  const v = +numEl.value;
  let pos = (v - lo) / (hi - lo) * SLIDER_STEPS;
  pos = Math.max(0, Math.min(SLIDER_STEPS, pos));
  sliderEl.value = Math.round(pos);
}

function setNumber(id, value) {
  const numEl = document.getElementById(`${id}-num`);
  if (!numEl) return;
  numEl.value = +value.toFixed(6);
  // Re-sync slider
  const sliderEl = document.getElementById(`${id}-slider`);
  if (numEl._linkedRange && sliderEl) {
    let pos = (value - numEl._linkedRange.lo) / (numEl._linkedRange.hi - numEl._linkedRange.lo) * SLIDER_STEPS;
    pos = Math.max(0, Math.min(SLIDER_STEPS, pos));
    sliderEl.value = Math.round(pos);
  }
}

function getNumber(id) {
  return +document.getElementById(`${id}-num`).value;
}

// ---------------------------------------------------------------------------
// Plot
// ---------------------------------------------------------------------------

function redrawPlot(highlight = null) {
  if (!currentDist) return;
  try {
    const dataJson = pyodide.runPython(
      `get_plot_data(${JSON.stringify(currentDist.name)}, ${JSON.stringify(currentParams)})`
    );
    const data = JSON.parse(dataJson);
    window._lastPlotData = data;
    drawPlotly(data, highlight);
    // Re-rescale prob x slider to new support
    rescaleSlider("prob-x", data.lo, data.hi);
    rescaleSlider("prob-b", data.lo, data.hi);
  } catch (e) {
    console.error("plot error:", e);
  }
}

function drawPlotly(data, highlight) {
  const { x, pdf, cdf, discrete, lo, hi } = data;

  const layoutCommon = {
    paper_bgcolor: "#1c1c1f",
    plot_bgcolor: "#1c1c1f",
    font: { color: "#e8e8ea", family: "system-ui, -apple-system, sans-serif", size: 12 },
    margin: { l: 50, r: 20, t: 30, b: 40 },
    xaxis: { gridcolor: "#2f2f33", zeroline: false, linecolor: "#555" },
    yaxis: { gridcolor: "#2f2f33", zeroline: false, linecolor: "#555" },
    showlegend: false,
    hovermode: "x unified",
  };

  // PDF/PMF subplot
  let pdfTraces = [];
  let cdfTraces = [];
  if (discrete) {
    pdfTraces.push({ x, y: pdf, type: "bar", marker: { color: "#319bff", line: { color: "#0a84ff", width: 1 } } });
    cdfTraces.push({ x, y: cdf, type: "scatter", mode: "lines", line: { color: "#319bff", width: 2, shape: "hv" } });
  } else {
    pdfTraces.push({ x, y: pdf, type: "scatter", mode: "lines", line: { color: "#319bff", width: 2 }, fill: "tozeroy", fillcolor: "rgba(49,155,255,0.15)" });
    cdfTraces.push({ x, y: cdf, type: "scatter", mode: "lines", line: { color: "#319bff", width: 2 } });
  }

  // Highlights based on current op
  if (currentOp === "probability") {
    const mode = document.getElementById("prob-mode").value;
    const xv = getNumber("prob-x");
    const bv = getNumber("prob-b");
    const hi_data = highlightForProbability(mode, xv, bv, x, pdf, cdf, discrete);
    if (hi_data) {
      pdfTraces.push(...hi_data.pdf);
      cdfTraces.push(...hi_data.cdf);
    }
  } else if (currentOp === "quantile") {
    const q = getNumber("quant-q");
    const xq = pyodide.runPython(`float(_build_rv(${JSON.stringify(currentDist.name)}, ${JSON.stringify(currentParams)})[1].ppf(${q}))`);
    const xv = +xq;
    const hi_data = highlightForProbability("le", xv, 0, x, pdf, cdf, discrete);
    if (hi_data) {
      pdfTraces.push(...hi_data.pdf);
      cdfTraces.push(...hi_data.cdf);
    }
  }

  Plotly.react("plot-pdf", pdfTraces, {
    ...layoutCommon,
    title: { text: discrete ? `${currentDist.name} — PMF` : `${currentDist.name} — PDF`, font: { size: 13 } },
    yaxis: { ...layoutCommon.yaxis, title: discrete ? "P(X = x)" : "f(x)" },
    xaxis: { ...layoutCommon.xaxis, title: "x" },
  }, { responsive: true, displayModeBar: false });

  Plotly.react("plot-cdf", cdfTraces, {
    ...layoutCommon,
    title: { text: "CDF", font: { size: 13 } },
    yaxis: { ...layoutCommon.yaxis, title: "P(X ≤ x)", range: [-0.02, 1.02] },
    xaxis: { ...layoutCommon.xaxis, title: "x" },
  }, { responsive: true, displayModeBar: false });
}

function highlightForProbability(mode, xv, bv, x, pdf, cdf, discrete) {
  const highlightColor = "#ff453a";
  const highlightFill = "rgba(255,69,58,0.4)";
  const pdfTraces = [];
  const cdfTraces = [];

  function sliceFill(xLo, xHi) {
    if (discrete) {
      const pts = x.map((xi, i) => ({ xi, p: pdf[i] })).filter(d => d.xi >= xLo && d.xi <= xHi);
      pdfTraces.push({
        x: pts.map(p => p.xi),
        y: pts.map(p => p.p),
        type: "bar",
        marker: { color: highlightColor, line: { color: "#ff6961", width: 1 } },
      });
    } else {
      const idxs = x.map((xi, i) => ({ xi, p: pdf[i], i })).filter(d => d.xi >= xLo && d.xi <= xHi);
      pdfTraces.push({
        x: idxs.map(p => p.xi),
        y: idxs.map(p => p.p),
        type: "scatter",
        mode: "lines",
        line: { color: highlightColor, width: 0 },
        fill: "tozeroy",
        fillcolor: highlightFill,
      });
    }
  }

  function vertical(xVal) {
    pdfTraces.push({ x: [xVal, xVal], y: [0, Math.max(...pdf) * 1.05], type: "scatter", mode: "lines", line: { color: "#888", width: 1, dash: "dash" } });
    cdfTraces.push({ x: [xVal, xVal], y: [0, 1], type: "scatter", mode: "lines", line: { color: "#888", width: 1, dash: "dash" } });
    cdfTraces.push({ x: [xVal], y: [interpCDF(xVal, x, cdf)], type: "scatter", mode: "markers", marker: { color: highlightColor, size: 8 } });
  }

  if (mode === "le" || mode === "lt") {
    sliceFill(-Infinity, mode === "lt" && discrete ? xv - 1 : xv);
    vertical(xv);
  } else if (mode === "ge" || mode === "gt") {
    sliceFill(mode === "gt" && !discrete ? xv : xv, Infinity);
    vertical(xv);
  } else if (mode === "between") {
    sliceFill(xv, bv);
    vertical(xv);
    vertical(bv);
  } else if (mode === "eq") {
    vertical(xv);
  }

  return { pdf: pdfTraces, cdf: cdfTraces };
}

function interpCDF(xv, xs, cdf) {
  for (let i = 0; i < xs.length - 1; i++) {
    if (xv >= xs[i] && xv <= xs[i + 1]) {
      const t = (xv - xs[i]) / (xs[i + 1] - xs[i] || 1);
      return cdf[i] + t * (cdf[i + 1] - cdf[i]);
    }
  }
  if (xv < xs[0]) return cdf[0];
  return cdf[cdf.length - 1];
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

function runProbability() {
  if (!currentDist) return;
  try {
    const mode = document.getElementById("prob-mode").value;
    const xv = getNumber("prob-x");
    const bv = getNumber("prob-b");
    const args = `${JSON.stringify(currentDist.name)}, ${JSON.stringify(currentParams)}, ${JSON.stringify(mode)}, ${xv}, ${bv}`;
    const p = +pyodide.runPython(`compute_probability(${args})`);
    document.getElementById("prob-result").textContent = p.toFixed(6);
    redrawPlot();
  } catch (e) {
    document.getElementById("prob-result").textContent = "error";
    console.error(e);
  }
}

function runQuantile() {
  if (!currentDist) return;
  try {
    const q = getNumber("quant-q");
    const x = +pyodide.runPython(`compute_quantile(${JSON.stringify(currentDist.name)}, ${JSON.stringify(currentParams)}, ${q})`);
    document.getElementById("quant-result").textContent = currentDist.discrete ? String(Math.round(x)) : x.toFixed(6);
    redrawPlot();
  } catch (e) {
    document.getElementById("quant-result").textContent = "error";
    console.error(e);
  }
}

function runSample() {
  if (!currentDist) return;
  try {
    const n = +document.getElementById("sample-n").value;
    const seed = +document.getElementById("sample-seed").value;
    const json = pyodide.runPython(`draw_sample(${JSON.stringify(currentDist.name)}, ${JSON.stringify(currentParams)}, ${n}, ${seed})`);
    const r = JSON.parse(json);
    const out = r.discrete ? r.values.join(", ") : r.values.map(v => v.toFixed(4)).join(", ");
    document.getElementById("sample-output").textContent = out;
    document.getElementById("sample-stats").innerHTML = `
      <div><span>Mean</span><b>${r.mean.toFixed(4)}</b></div>
      <div><span>SD</span><b>${r.sd.toFixed(4)}</b></div>
      <div><span>Min</span><b>${r.min.toFixed(4)}</b></div>
      <div><span>Max</span><b>${r.max.toFixed(4)}</b></div>
    `;
  } catch (e) {
    document.getElementById("sample-output").textContent = `error: ${e.message || e}`;
    console.error(e);
  }
}

function runSolve() {
  if (!currentDist) return;
  const resultEl = document.getElementById("solve-result");
  try {
    const param = document.getElementById("solve-param").value;
    const xv = +document.getElementById("solve-x").value;
    const tp = +document.getElementById("solve-p").value;
    const lo = +document.getElementById("solve-lo").value;
    const hi = +document.getElementById("solve-hi").value;
    const json = pyodide.runPython(
      `solve_parameter(${JSON.stringify(currentDist.name)}, ${JSON.stringify(currentParams)}, ${JSON.stringify(param)}, ${xv}, ${tp}, ${lo}, ${hi})`
    );
    const r = JSON.parse(json);
    if (r.integer_search) {
      resultEl.textContent = `${param} = ${r.value}  (Δ=${r.error.toExponential(2)})`;
    } else {
      resultEl.textContent = `${param} = ${r.value.toFixed(6)}`;
    }
  } catch (e) {
    resultEl.textContent = "error";
    const errMsg = e.message || String(e);
    // Show a tooltip-like message
    const note = document.createElement("div");
    note.className = "error";
    note.textContent = errMsg.split("\n").filter(l => l.includes("ValueError") || l.includes("error")).pop() || errMsg;
    const panel = document.querySelector('.op-panel[data-op="solve"]');
    panel.querySelectorAll(".error").forEach(n => n.remove());
    panel.appendChild(note);
    console.error(e);
  }
}

// ---------------------------------------------------------------------------
// Utils
// ---------------------------------------------------------------------------

function escapeAttr(s) { return String(s).replace(/"/g, "&quot;"); }
function cssId(s) { return s.replace(/[^a-zA-Z0-9_-]/g, "_"); }

// Go!
init();
